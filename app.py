
# app.py
import asyncio
import builtins

from langgraph.types import Command

from main_graph.graph_builder import graph
from main_graph.graph_states import InputState
from utils.utils import new_uuid

thread = {"configurable": {"thread_id": new_uuid()}}

async def process_query(query):
    input_state = InputState(messages=query)
    try:
        async for c, metadata in graph.astream(input=input_state, stream_mode="messages", config=thread):
            if c.additional_kwargs.get("tool_calls"):
                print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="", flush=True)
            if c.content:
                print(c.content, end="", flush=True)
                await asyncio.sleep(0.1)  # Small delay for better streaming

            if hasattr(c, 'interrupts') and c.interrupts:
                prompt = c.interrupts["question"]
                llm_output = c.interrupts["llm_output"]
                print(f"\nLLM Output: {llm_output}")
                while True:  # Loop for input validation
                    response = input(f"\n{prompt} (y/n): ").strip().lower()
                    if response in ('y', 'n'):
                        break
                    print("Invalid input. Please enter 'y' or 'n'.")

                if response == 'y':
                    # Resume with user's response
                    async for resumed_c, _ in graph.astream(Command(resume=response), stream_mode="messages",
                                                            config=thread):
                        if resumed_c.additional_kwargs.get("tool_calls"):
                            print(resumed_c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="",
                                flush=True)
                        if resumed_c.content:
                            print(resumed_c.content, end="", flush=True)
                            await asyncio.sleep(0.1)
                else:
                    print("Ending process.")
                    return  # Exit the function

    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    print("Enter your query (type '-q' to quit):")
    while True:
        query = input("> ").strip()
        if query.lower() == "-q":
            print("Exiting...")
            break
        await process_query(query)


if __name__ == "__main__":
    asyncio.run(main())
    
