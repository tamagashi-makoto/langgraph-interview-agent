import os
import asyncio
from typing import Annotated, Literal, TypedDict, Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import (
    ToolMessage, HumanMessage, BaseMessage, AIMessage, SystemMessage
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_prompt_optimizer

# ==========================================
# Settings and Constants
# ==========================================

# Memory (InMemoryStore for now, Redis or Postgres is expected in production)
store = InMemoryStore(
    index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
)

# Job description data (for RAG)
JOB_DESCRIPTION_DOCS = {
    "tech_stack": "Python (FastAPI, Django), TypeScript (React/Vue), AWS, Docker, Kubernetes",
    "required_experience": "Web application development experience 3 years or more, or AI model implementation experience",
    "soft_skills": "Team coordination, ability to express technical selection reasons in language, self-motivation",
    "culture": "Startup mindset. Evaluate the ability to make decisions in uncertain situations and move forward.",
    "salary_range": "800万〜1200万円 (Skill-based salary)",
}

# ==========================================
# Type Definitions
# ==========================================

class Triple(BaseModel):
    subject: str = Field(description="Subject (e.g: Candidate)")
    predicate: str = Field(description="Predicate (e.g: Experience)")
    object: str = Field(description="Object (e.g: Python Development for 5 years)")
    context: str | None = Field(default=None, description="Context")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    last_agent: Literal["interview_agent", "review_agent"]
    loop_count: int

# ==========================================
# Graph Class Definition
# ==========================================

class SessionGraph():
    def __init__(self, config: RunnableConfig):
        self.config = config
        self.is_langgraph_dev = os.getenv("IS_LANGGRAPH_DEV") == "1"
        
        # Prompt optimizer
        self.prompt_optimizer = create_prompt_optimizer(
            'openai:gpt-4o-mini',
            kind='gradient',
            config={'max_reflection_steps': 2, 'min_reflection_steps': 1},
        )
        
        self._interview_agent = self._make_interview_agent()
        self._review_agent = self._make_review_agent()
        
        # Queue to receive input from outside
        self.human_message_queue: asyncio.Queue[str|None] = asyncio.Queue()

    async def __aenter__(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("interview_agent", self._call_interview_agent)
        graph_builder.add_node("review_agent", self._call_review_agent)
        graph_builder.add_node("human", self._human_node_in_dev if self.is_langgraph_dev else self._human_node)

        graph_builder.set_entry_point("interview_agent")
        
        # For demo, use In-Memory checkpoint
        self._graph = graph_builder.compile(checkpointer=MemorySaver())
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return

    # ------------------------------------------
    # Tools
    # ------------------------------------------

    @staticmethod
    def make_handoff_tool(*, agent_name: str):
        tool_name = f"transfer_to_{agent_name}"
        @tool(tool_name)
        def handoff_to_agent(
            reason: str,
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ):
            """Transfer task to another agent"""
            tool_msg = ToolMessage(
                content=f"Transfer to {agent_name}. Reason: {reason}",
                tool_call_id=tool_call_id,
                name=tool_name,
            )
            return Command(
                goto=agent_name,
                graph=Command.PARENT,
                update={"messages": state["messages"] + [tool_msg]},
            )
        return handoff_to_agent

    @staticmethod
    @tool
    def retrieve_job_requirements(query: str):
        """Search job description documents (job description, evaluation criteria)"""
        results = []
        for key, value in JOB_DESCRIPTION_DOCS.items():
            if query.lower() in key or query.lower() in value.lower() or "全般" in query:
                results.append(f"【{key}】: {value}")
        if not results:
            return "該当する情報はドキュメントに見つかりませんでした。"
        return "\n".join(results)

    # ------------------------------------------
    # Agent Definition
    # ------------------------------------------

    def _make_interview_agent(self):
        manage_memory_tool = create_manage_memory_tool(
            namespace=("candidate_profile",),
            instructions="Save candidate information",
            schema=Triple,
            store=store,
        )
        tools = [
            self.retrieve_job_requirements,
            manage_memory_tool,
            self.make_handoff_tool(agent_name="review_agent"),
        ]
        system_prompt = """
        You are an AI company's recruitment engineer (interviewer).
        1. First, check the requirements using `retrieve_job_requirements`.
        2. Ask the candidate questions to confirm their skills.
        3. If the information is sufficient or if there is an instruction from the reviewer, call `transfer_to_review_agent`.
        """
        return create_react_agent("openai:gpt-4o-mini", tools=tools, name="Interview Agent", state_modifier=system_prompt)

    async def _call_interview_agent(self, state: State) -> Command:
        # For debugging, print to standard output (can be removed or minimized in production)
        # print("DEBUG: Interview Agent Active")
        response = await self._interview_agent.ainvoke(state)
        return Command(update={**response, "last_agent": "interview_agent"}, goto="human")

    def _make_review_agent(self):
        tools = [self.make_handoff_tool(agent_name="interview_agent")]
        base_prompt = """
        You are a senior engineer who makes the first judgment on whether to hire or not.
        Analyze the interview log.
        - If there is insufficient information, use `transfer_to_interview_agent` to give specific instructions and return.
        - If it is sufficient, output the approval or rejection and end.
        """
        return create_react_agent("openai:gpt-4o-mini", tools=tools, name="Review Agent", state_modifier=base_prompt)

    async def _call_review_agent(self, state: State) -> Command:
        messages = state.get("messages", [])
        loop_count = state.get("loop_count", 0)

        # Guardrail
        if loop_count > 3:
            return Command(
                update={"messages": [AIMessage(content="Review process is taking too long. Terminating.")]},
                goto="__end__"
            )

        # Prompt optimization
        trajectories = self._build_trajectories_from_messages(messages)
        final_system_message_content = state.get("messages")[0].content if state.get("messages") else ""

        if trajectories:
            base_prompt_for_opt = "You are a recruitment engineer. Use the conversation history to improve the prompt so that the analysis is more accurate."
            try:
                opt_res = await self.prompt_optimizer.ainvoke({'trajectories': trajectories, 'prompt': base_prompt_for_opt})
                optimized = opt_res.get('optimized_prompt')
                # Inject
                final_system_message_content = f"{optimized}\n\n(Basic instructions)\nInformation is sufficient? If not, return to the interviewer."
            except Exception:
                pass

        modified_messages = [SystemMessage(content=final_system_message_content)] + messages
        response = await self._review_agent.ainvoke({**state, "messages": modified_messages})
        
        last_msg = response["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return Command(
                update={**response, "last_agent": "review_agent", "loop_count": loop_count + 1},
                goto="interview_agent"
            )
        
        return Command(update={**response, "last_agent": "review_agent"}, goto="__end__")

    def _build_trajectories_from_messages(self, messages):
        trajectories = []
        current_pair = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                current_pair.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                current_pair.append({"role": "assistant", "content": msg.content})
            if len(current_pair) == 2:
                trajectories.append((current_pair, None))
                current_pair = []
        return trajectories

    async def _human_node(self, state: State) -> Command:
        human_message = ""
        while True:
            chunk = await self.human_message_queue.get()
            if chunk is None: break
            human_message += chunk
        return Command(update={"messages": [HumanMessage(content=human_message)]}, goto="interview_agent")
    
    async def _human_node_in_dev(self, state: State):
        # dev mode dummy
        return Command(goto="interview_agent")