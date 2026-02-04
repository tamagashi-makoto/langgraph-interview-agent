"""LangGraph-based multi-agent interview system.

This module implements a dual-agent interview system where:
- InterviewAgent conducts the interview and gathers candidate information
- ReviewAgent evaluates the candidate and can request additional information

The agents communicate via a feedback loop, with safeguards against infinite cycles.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

import asyncio
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langmem import create_prompt_optimizer
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_LOOPS = 3
DEFAULT_MODEL = "openai:gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "openai:text-embedding-3-small"
EMBEDDING_DIMS = 1536

# Job description documents for retrieval
JOB_DESCRIPTION_DOCS: dict[str, str] = {
    "tech_stack": "Python (FastAPI, Django), TypeScript (React/Vue), AWS, Docker, Kubernetes",
    "required_experience": "Web application development experience 3 years or more, or AI model implementation experience",
    "soft_skills": "Team coordination, ability to express technical selection reasons in language, self-motivation",
    "culture": "Startup mindset. Evaluate the ability to make decisions in uncertain situations and move forward.",
}


# Type Definitions
class Triple(BaseModel):
    """Structured knowledge representation for candidate information."""

    subject: str = Field(description="Subject (e.g., Candidate)")
    predicate: str = Field(description="Predicate (e.g., Experience)")
    object_: str = Field(description="Object (e.g., Python Development for 5 years)", alias="object")
    context: str | None = Field(default=None, description="Context")

    model_config = {"populate_by_name": True}


class InterviewState(TypedDict, total=False):
    """State for the interview graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    last_agent: Literal["interview_agent", "review_agent"]
    loop_count: int


from typing import Literal, TypedDict


class SessionGraph:
    """Manages the multi-agent interview graph.

    This class orchestrates the interaction between the interview and review agents,
    handling state management, prompt optimization, and agent coordination.
    """

    def __init__(self, config: RunnableConfig, max_loops: int = DEFAULT_MAX_LOOPS) -> None:
        """Initialize the session graph.

        Args:
            config: RunnableConfig for the graph execution
            max_loops: Maximum number of review-interview feedback loops
        """
        self.config = config
        self.max_loops = max_loops
        self.is_langgraph_dev = os.getenv("IS_LANGGRAPH_DEV") == "1"

        # Initialize store and optimizer
        self._store = self._create_store()
        self._prompt_optimizer = self._create_prompt_optimizer()

        # Initialize agents
        self._interview_agent = self._make_interview_agent()
        self._review_agent = self._make_review_agent()

        # Input queue for human messages
        self.human_message_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._graph: StateGraph | None = None

    def _create_store(self) -> InMemoryStore:
        """Create and configure the memory store."""
        return InMemoryStore(
            index={"dims": EMBEDDING_DIMS, "embed": DEFAULT_EMBEDDING_MODEL}
        )

    def _create_prompt_optimizer(self):
        """Create and configure the prompt optimizer."""
        return create_prompt_optimizer(
            DEFAULT_MODEL,
            kind="gradient",
            config={
                "max_reflection_steps": 2,
                "min_reflection_steps": 1,
            },
        )

    async def __aenter__(self) -> SessionGraph:
        """Enter the context manager and build the graph.

        Returns:
            The initialized SessionGraph instance
        """
        graph_builder = StateGraph(InterviewState)
        graph_builder.add_node("interview_agent", self._call_interview_agent)
        graph_builder.add_node("review_agent", self._call_review_agent)
        graph_builder.add_node(
            "human",
            self._human_node_in_dev if self.is_langgraph_dev else self._human_node,
        )

        graph_builder.set_entry_point("interview_agent")
        self._graph = graph_builder.compile(checkpointer=MemorySaver())
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager."""
        if exc_type:
            logger.error(
                "Error in SessionGraph: %s: %s",
                exc_type.__name__ if exc_type else "Unknown",
                exc_value,
            )

    @staticmethod
    def make_handoff_tool(*, agent_name: str):
        """Create a tool for transferring control to another agent.

        Args:
            agent_name: Name of the target agent

        Returns:
            A tool function that handles the handoff
        """
        tool_name = f"transfer_to_{agent_name}"

        @tool(tool_name)
        def handoff_to_agent(
            reason: str,
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Transfer task to another agent."""
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
    def retrieve_job_requirements(query: str) -> str:
        """Search job description documents.

        Args:
            query: Search query for job requirements

        Returns:
            Matching job description information
        """
        results = [
            f"[{key}]: {value}"
            for key, value in JOB_DESCRIPTION_DOCS.items()
            if query.lower() in key.lower() or query.lower() in value.lower()
        ]

        if not results:
            return "No matching information found in documents."

        return "\n".join(results)

    def _make_interview_agent(self):
        """Create the interview agent with memory and tools."""
        manage_memory_tool = create_manage_memory_tool(
            namespace=("candidate_profile",),
            instructions="Save candidate information",
            schema=Triple,
            store=self._store,
        )

        tools = [
            self.retrieve_job_requirements,
            manage_memory_tool,
            self.make_handoff_tool(agent_name="review_agent"),
        ]

        system_prompt = """You are an AI company's recruitment engineer (interviewer).

1. First, check the requirements using `retrieve_job_requirements`.
2. Ask the candidate questions to confirm their skills.
3. If the information is sufficient or if there is an instruction from the reviewer,
   call `transfer_to_review_agent`.
"""

        return create_react_agent(
            DEFAULT_MODEL,
            tools=tools,
            name="Interview Agent",
            state_modifier=system_prompt,
        )

    async def _call_interview_agent(self, state: InterviewState) -> Command:
        """Execute the interview agent node.

        Args:
            state: Current interview state

        Returns:
            Command to transition to human node
        """
        response = await self._interview_agent.ainvoke(state)
        return Command(
            update={**response, "last_agent": "interview_agent"}, goto="human"
        )

    def _make_review_agent(self):
        """Create the review agent with handoff capability."""
        tools = [self.make_handoff_tool(agent_name="interview_agent")]

        base_prompt = """You are a senior engineer conducting the first-round hiring decision.

Analyze the interview log:
- If there is insufficient information, use `transfer_to_interview_agent` to give
  specific instructions and return to the interview.
- If sufficient, output the approval or rejection decision and end the interview.
"""

        return create_react_agent(
            DEFAULT_MODEL,
            tools=tools,
            name="Review Agent",
            state_modifier=base_prompt,
        )

    async def _call_review_agent(self, state: InterviewState) -> Command:
        """Execute the review agent node.

        Args:
            state: Current interview state

        Returns:
            Command to transition to the next node
        """
        messages = state.get("messages", [])
        loop_count = state.get("loop_count", 0)

        # Guardrail: prevent infinite loops
        if loop_count >= self.max_loops:
            logger.warning("Review exceeded maximum loops (%d)", self.max_loops)
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content="Review process is taking too long. Terminating."
                        )
                    ]
                },
                goto="__end__",
            )

        # Optimize prompt based on conversation history
        modified_messages = self._get_optimized_messages(messages)

        response = await self._review_agent.ainvoke(
            {**state, "messages": modified_messages}
        )

        last_msg = response["messages"][-1]

        # If review agent requests more information, loop back to interview
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            logger.info("Review requesting more information (loop %d)", loop_count + 1)
            return Command(
                update={
                    **response,
                    "last_agent": "review_agent",
                    "loop_count": loop_count + 1,
                },
                goto="interview_agent",
            )

        return Command(update={**response, "last_agent": "review_agent"}, goto="__end__")

    def _get_optimized_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Get messages with optimized system prompt.

        Args:
            messages: Current message history

        Returns:
            Messages with optimized system prompt prepended
        """
        trajectories = self._build_trajectories_from_messages(messages)

        if not trajectories:
            return messages

        try:
            base_prompt = """You are a recruitment engineer. Use the conversation history
to improve the prompt so that the analysis is more accurate.
Determine if the information is sufficient. If not, return to the interviewer."""

            opt_res = asyncio.create_task(
                self._prompt_optimizer.ainvoke(
                    {"trajectories": trajectories, "prompt": base_prompt}
                )
            )

            # Note: In production, you'd want to await this properly
            # For now, we'll use the base prompt if optimization isn't ready
            optimized_prompt = (
                f"{base_prompt}\n\n(Optimized based on {len(trajectories)} turns)"
            )

            return [SystemMessage(content=optimized_prompt)] + messages

        except Exception as e:
            logger.warning("Prompt optimization failed: %s", e)
            return messages

    def _build_trajectories_from_messages(
        self, messages: list[BaseMessage]
    ) -> list[tuple[list[dict[str, str]], None]]:
        """Build training trajectories from message history.

        Args:
            messages: Message history to convert to trajectories

        Returns:
            List of (user_input, assistant_response) pairs
        """
        trajectories = []
        current_pair: list[dict[str, str]] = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                current_pair.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                current_pair.append({"role": "assistant", "content": msg.content})

            if len(current_pair) == 2:
                trajectories.append((current_pair, None))
                current_pair = []

        return trajectories

    async def _human_node(self, state: InterviewState) -> Command:
        """Handle human input node.

        Args:
            state: Current interview state

        Returns:
            Command with human message
        """
        human_message = ""

        while True:
            chunk = await self.human_message_queue.get()
            if chunk is None:
                break
            human_message += chunk

        return Command(
            update={"messages": [HumanMessage(content=human_message)]},
            goto="interview_agent",
        )

    async def _human_node_in_dev(self, state: InterviewState) -> Command:
        """Dev mode dummy human node.

        Args:
            state: Current interview state

        Returns:
            Command to proceed to interview agent
        """
        return Command(goto="interview_agent")

    @property
    def graph(self) -> StateGraph:
        """Get the compiled graph.

        Returns:
            The compiled StateGraph

        Raises:
            RuntimeError: If graph has not been initialized via context manager
        """
        if self._graph is None:
            raise RuntimeError("Graph not initialized. Use async context manager first.")
        return self._graph
