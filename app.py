from subgraph.graph_states import ResearcherState
from main_graph.graph_states import AgentState
from utils.utils import config, new_uuid
from subgraph.graph_builder import researcher_graph
from main_graph.graph_builder import InputState, graph
from langgraph.types import Command
import asyncio
import uuid
import os
import time
import builtins

thread = {"configurable": {"thread_id": new_uuid()}}
#This is a question related to environmental context. tell me the data center PUE efficiency value in Dublin in 2021

async def process_query(query):
    inputState = InputState(messages=query)

    async for c, metadata in graph.astream(input=inputState, stream_mode="messages", config=thread):
        if c.additional_kwargs.get("tool_calls"):
            print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="", flush=True)
        if c.content:
            await asyncio.sleep(0.5)  # Non-blocking delay between chunks.
            print(c.content, end="", flush=True)
            
    # Add a 60-second delay between steps.
    await asyncio.sleep(60)

    # Step 2: Check for interrupts in the graph's state.
    current_state = graph.get_state(thread)
    if current_state and current_state[-1] and current_state[-1][0].interrupts:
        response = input("\nThe response may contain uncertain information. Retry the generation? If yes, press 'y': ")
        if response.lower() == 'y':
            async for c, metadata in graph.astream(Command(resume=response), stream_mode="messages", config=thread):
                if c.additional_kwargs.get("tool_calls"):
                    print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="", flush=True)
                if c.content:
                    await asyncio.sleep(0.5)
                    print(c.content, end="", flush=True)
    
    # Optionally, add a delay after processing the query.
    await asyncio.sleep(60)

async def main():
    input = builtins.input
    print("Enter your query (type '-q' to quit):")
    while True:
        query = input("> ")
        if query.strip().lower() == "-q":
            print("Exiting...")
            break
        await process_query(query)
        # Add a delay between processing different queries.
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
