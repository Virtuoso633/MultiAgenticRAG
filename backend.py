# backend.py

import asyncio
from typing import AsyncGenerator
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command

from main_graph.graph_builder import graph  # Import your LangGraph
from main_graph.graph_states import InputState
from utils.utils import new_uuid
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
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

# In backend.py, update the process_graph_stream function

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

            # Handle interruptions - UPDATED CODE
            try:
                # Check if this is an Interrupt instance
                if hasattr(chunk, '_key') and chunk._key == "interrupt":
                    interrupt_data = chunk.value
                    prompt = interrupt_data.get("question", "Do you want to retry the generation? (y/n)")
                    llm_output = interrupt_data.get("llm_output", "No output available")
                    binary_score = chunk.hallucination.binary_score if hasattr(chunk, "hallucination") and chunk.hallucination else "0"

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
                    return  # Stop processing until frontend responds
                
                # Alternative approach - check for interrupts field
                elif hasattr(chunk, 'interrupts') and chunk.interrupts:
                    prompt = chunk.interrupts.get("question", "Do you want to retry the generation? (y/n)")
                    llm_output = chunk.interrupts.get("llm_output", "No output available")
                    binary_score = chunk.hallucination.binary_score if hasattr(chunk, "hallucination") and chunk.hallucination else "0"

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
                    return  # Stop processing until frontend responds
                    
            except Exception as e:
                logger.error(f"Error handling interrupt: {e}")

        await websocket.send_json({"end": True})  # Completion

    except Exception as e:
        logger.error(f"Error in process_graph_stream: {e}")
        await websocket.send_json({"error": str(e)})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = new_uuid()  # Assign a unique client ID
    await websocket.accept()
    logger.info("connection open")
    
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
