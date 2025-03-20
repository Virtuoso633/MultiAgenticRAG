# backend.py

import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command

from main_graph.graph_builder import graph  # Import your LangGraph
from main_graph.graph_states import InputState
from utils.utils import new_uuid

app = FastAPI()

# CORS (Cross-Origin Resource Sharing) setup: Allow requests from your frontend
origins = [
    "http://localhost:3000",  # Allow requests from your React app
    "http://localhost:8000",   # Add if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store pending interruptions per client
pending_interruptions = {}


async def process_graph_stream(query: str, websocket: WebSocket, client_id: str) -> AsyncGenerator:
    """Processes a query using the LangGraph and streams results."""
    
    input_state = InputState(messages=[query])  # Create InputState directly
    thread_id = new_uuid()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        async for chunk, _ in graph.astream(input_state, stream_mode="messages", config=config):
            
            if chunk.additional_kwargs.get("tool_calls"):
                tool_calls = chunk.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
                await websocket.send_json({"tool_calls": tool_calls})

            if hasattr(chunk, 'queries') and chunk.queries:  # Handle queries
                await websocket.send_json({"queries": chunk.queries})

            if chunk.content:
                await websocket.send_json({"content": chunk.content})

            await asyncio.sleep(0.05)

            # Handle interruptions
            if hasattr(chunk, 'interrupts') and chunk.interrupts:
                prompt = chunk.interrupts["question"]
                llm_output = chunk.interrupts["llm_output"]
                binary_score = chunk.hallucination.binary_score if chunk.hallucination else "N/A"  # Safe access

                # Store interruption state
                pending_interruptions[client_id] = {
                    "thread_id": thread_id,
                    "config": config
                }

                await websocket.send_json({
                    "interrupt": {
                        "question": prompt,
                        "llm_output": llm_output,
                        "binary_score": binary_score
                    }
                })

                return  # Stop processing until frontend sends a response

        await websocket.send_json({"end": True})  # Completion

    except Exception as e:
        await websocket.send_json({"error": str(e)})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = new_uuid()  # Assign a unique client ID
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if isinstance(data, dict) and "query" in data:
                asyncio.create_task(process_graph_stream(data["query"], websocket, client_id))

            elif isinstance(data, dict) and "resume" in data:  # Handle frontend's "y/n" response
                if client_id in pending_interruptions:
                    thread_id = pending_interruptions[client_id]["thread_id"]
                    config = pending_interruptions[client_id]["config"]
                    
                    del pending_interruptions[client_id]  # Remove pending state

                    if data["resume"].lower() == "y":
                        async for resumed_chunk, _ in graph.astream(Command(resume="y"), stream_mode="messages", config=config):
                            
                            if resumed_chunk.additional_kwargs.get("tool_calls"):
                                tool_calls_resumed = resumed_chunk.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
                                await websocket.send_json({"tool_calls": tool_calls_resumed})

                            if hasattr(resumed_chunk, 'queries') and resumed_chunk.queries:
                                await websocket.send_json({"queries": resumed_chunk.queries})

                            if resumed_chunk.content:
                                await websocket.send_json({"content": resumed_chunk.content})

                            await asyncio.sleep(0.05)

                    else:
                        await websocket.send_json({"end": True})  # Signal end
                        return  # Exit

    except WebSocketDisconnect:
        print("Client disconnected")
        
        if client_id in pending_interruptions:
            del pending_interruptions[client_id]
    
    except Exception as e:
        print(f"Error: {e}")


@app.get("/")
async def read_root():
    return {"message": "MultiAgentic RAG Backend is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
