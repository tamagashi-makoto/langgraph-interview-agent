"""Demo CLI for the multi-agent interview system.

This module provides an interactive command-line interface for running
the interview and review agent system.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum

import colorama
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agent_core import SessionGraph

# Configure module logging
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Agent types for output formatting."""

    INTERVIEWER = "Interviewer"
    REVIEWER = "Reviewer"
    UNKNOWN = "Unknown"

    @classmethod
    def from_name(cls, name: str | None) -> AgentType:
        """Determine agent type from message name.

        Args:
            name: The agent name from the message

        Returns:
            The corresponding AgentType enum value
        """
        if not name:
            return AgentType.UNKNOWN

        name_lower = name.lower()
        if "interview" in name_lower:
            return AgentType.INTERVIEWER
        if "review" in name_lower:
            return AgentType.REVIEWER
        return AgentType.UNKNOWN


class ConsoleColors:
    """ANSI color codes for terminal output."""

    INTERVIEWER = colorama.Fore.CYAN
    REVIEWER = colorama.Fore.MAGENTA
    SYSTEM = colorama.Fore.YELLOW
    USER = colorama.Fore.GREEN
    RESET = colorama.Style.RESET_ALL


class InterviewRunner:
    """Manages the execution of the interview demo."""

    # Exit commands
    EXIT_COMMANDS = {"quit", "exit", "end", "q"}

    # Keywords that indicate a final decision has been made
    DECISION_KEYWORDS = {"success", "fail", "approved", "rejected", "hire", "pass"}

    def __init__(self, config: RunnableConfig | None = None) -> None:
        """Initialize the interview runner.

        Args:
            config: Optional RunnableConfig for the graph execution
        """
        self.config = config or RunnableConfig(
            {"configurable": {"thread_id": "demo_thread"}}
        )
        self._should_stop = False

    def _print_banner(self) -> None:
        """Print the demo banner."""
        banner = (
            f"{colorama.Back.WHITE}{colorama.Fore.BLACK}"
            " === AI Interviewer & Reviewer Multi-Agent Demo === "
            f"{ConsoleColors.RESET}"
        )
        print(banner)
        print(f"[{ConsoleColors.INTERVIEWER}Interviewer{ConsoleColors.RESET}] Python Engineer Interview")
        print(f"[{ConsoleColors.REVIEWER}Reviewer{ConsoleColors.RESET}] Waiting in the background...\n")

    def _print_agent_message(self, msg: AIMessage, agent_type: AgentType) -> None:
        """Print an agent message with appropriate coloring.

        Args:
            msg: The AI message to print
            agent_type: The type of agent sending the message
        """
        color = (
            ConsoleColors.INTERVIEWER
            if agent_type == AgentType.INTERVIEWER
            else ConsoleColors.REVIEWER
        )

        if msg.tool_calls:
            tool_names = ", ".join([t["name"] for t in msg.tool_calls])
            print(f"{color}   (tool: {tool_names} running...){ConsoleColors.RESET}")
        elif msg.content:
            print(f"\n{color}{agent_type.value}: {msg.content}{ConsoleColors.RESET}")

    def _check_decision_made(self, content: str) -> bool:
        """Check if a final decision has been made.

        Args:
            content: The message content to check

        Returns:
            True if decision keywords are present
        """
        return any(keyword in content.lower() for keyword in self.DECISION_KEYWORDS)

    async def _input_loop(self, session: SessionGraph) -> None:
        """Handle user input in an async loop.

        Args:
            session: The active SessionGraph instance
        """
        while not self._should_stop:
            try:
                text = await asyncio.to_thread(
                    input,
                    f"\n{ConsoleColors.USER}You (answer): {ConsoleColors.RESET}"
                )

                if text.lower().strip() in self.EXIT_COMMANDS:
                    print("Demo ended.")
                    self._should_stop = True
                    await session.human_message_queue.put(None)
                    break

                await session.human_message_queue.put(text)

            except EOFError:
                print("\nInput ended.")
                self._should_stop = True
                await session.human_message_queue.put(None)
                break

    async def run(self) -> None:
        """Run the interview demo."""
        colorama.init(autoreset=True)
        self._print_banner()
        print(f"{ConsoleColors.SYSTEM}--- Interview started ---{ConsoleColors.RESET}")

        try:
            async with SessionGraph(self.config) as session:
                input_task = asyncio.create_task(self._input_loop(session))

                async for event in session.graph.astream(
                    {"messages": [], "loop_count": 0},
                    config=self.config,
                    stream_mode="values",
                ):
                    if self._should_stop:
                        break

                    messages = event.get("messages", [])
                    if not messages:
                        continue

                    last_msg = messages[-1]

                    if isinstance(last_msg, AIMessage):
                        agent_type = AgentType.from_name(last_msg.name)
                        self._print_agent_message(last_msg, agent_type)

                        # Check if a final decision was made
                        if last_msg.content and self._check_decision_made(last_msg.content):
                            print(
                                f"\n{ConsoleColors.SYSTEM}"
                                "=== Decision reached, ending interview ==="
                                f"{ConsoleColors.RESET}"
                            )
                            self._should_stop = True
                            input_task.cancel()
                            break

                # Wait for input task to complete
                try:
                    await input_task
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            logger.info("Demo cancelled by user")
        except Exception as e:
            logger.error("Error running demo: %s", e, exc_info=True)
            print(f"{ConsoleColors.SYSTEM}Error: {e}{ConsoleColors.RESET}")


async def run_demo() -> None:
    """Entry point for the demo application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    runner = InterviewRunner()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(run_demo())
