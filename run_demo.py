import asyncio
import colorama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent_core import SessionGraph

colorama.init(autoreset=True)
COL_INTERVIEWER = colorama.Fore.CYAN     # Interviewer (blue)
COL_REVIEWER = colorama.Fore.MAGENTA     # Reviewer (purple)
COL_SYSTEM = colorama.Fore.YELLOW        # System notification (yellow)
COL_USER = colorama.Fore.GREEN           # User (green)
COL_RESET = colorama.Style.RESET_ALL

async def run_demo():
    print(f"{colorama.Back.WHITE}{colorama.Fore.BLACK} === AI Interviewer & Reviewer Multi-Agent Demo === {COL_RESET}")
    print("※ Run `agent_core.py` to import and execute.\n")
    print(f"[{COL_INTERVIEWER}Interviewer{COL_RESET}] Python Engineer Interview.")
    print(f"[{COL_REVIEWER}Reviewer{COL_RESET}] Waiting in the background...\n")

    config = RunnableConfig({"configurable": {"thread_id": "demo_thread_v2"}})
    
    async with SessionGraph(config) as session:
        graph = session._graph
        
        async def input_loop():
            while True:
                text = await asyncio.to_thread(input, f"\n{COL_USER}You (answer): {COL_RESET}")
                if text.lower() in ["quit", "exit", "end"]:
                    print("Demo ended.")
                    break
                await session.human_message_queue.put(text)

        input_task = asyncio.create_task(input_loop())

        print(f"{COL_SYSTEM}--- Interview started ---{COL_RESET}")
        
        try:
            async for event in graph.astream(
                {"messages": [], "loop_count": 0},
                config=config,
                stream_mode="values"
            ):
                messages = event.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]
                
                if isinstance(last_msg, AIMessage):
                    content = last_msg.content
                    sender_name = last_msg.name or "unknown"
                    
                    # ツール呼び出し（思考中）の表示
                    if not content and last_msg.tool_calls:
                        tool_names = ", ".join([t["name"] for t in last_msg.tool_calls])
                        # 誰がツールを呼んだかで色を変える
                        if "Interviewer" in sender_name or "interview" in sender_name:
                            print(f"{COL_INTERVIEWER}   (tool: {tool_names} is running...){COL_RESET}")
                        elif "Reviewer" in sender_name or "review" in sender_name:
                            print(f"{COL_REVIEWER}   (tool: {tool_names} is running...){COL_RESET}")
                        continue

                    # エージェントごとの発言表示
                    if "Interviewer" in sender_name or "interview" in sender_name:
                        print(f"\n{COL_INTERVIEWER}Interviewer: {content}{COL_RESET}")
                    
                    elif "Reviewer" in sender_name or "review" in sender_name:
                        print(f"\n{COL_REVIEWER}Reviewer: {content}{COL_RESET}")
                        
                        # 合否判定が出たら終了
                        if "success" in content or "fail" in content:
                            print(f"\n{COL_SYSTEM}=== The decision was made, ending the interview ==={COL_RESET}")
                            input_task.cancel()
                            return

            await input_task
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_demo())