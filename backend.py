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

# async def process_graph_stream(query: str, websocket: WebSocket, client_id: str):
#     """Processes a query using the LangGraph and streams results."""
    
#     input_state = InputState(messages=[query])
#     thread_id = new_uuid()
#     config = {"configurable": {"thread_id": thread_id}}

#     try:
#         async for chunk, _ in graph.astream(input_state, stream_mode="messages", config=config):
            
#             if chunk.additional_kwargs.get("tool_calls"):
#                 tool_calls = chunk.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
#                 await websocket.send_json({"tool_calls": tool_calls})

#             if hasattr(chunk, 'queries') and chunk.queries:
#                 await websocket.send_json({"queries": chunk.queries})

#             if chunk.content:
#                 await websocket.send_json({"content": chunk.content})

#             await asyncio.sleep(0.05)


#             # Handle interruptions by delegating approval to the frontend
#             if hasattr(chunk, 'interrupts') and chunk.interrupts:
#                 prompt = chunk.interrupts["question"]
#                 llm_output = chunk.interrupts["llm_output"]
#                 binary_score = chunk.hallucination.binary_score if chunk.hallucination else "N/A"
                
#                 # Create a Future for the human response and store it using the client_id
#                 resume_future = asyncio.Future()
#                 pending_interruptions[client_id] = resume_future

#                 # Send the interrupt data to the frontend UI
#                 await websocket.send_json({
#                     "interrupt": {
#                         "question": prompt,
#                         "llm_output": llm_output,
#                         "binary_score": binary_score
#                     }
#                 })

#                 # Wait for the frontend's resume decision (it will send "y" or "n")
#                 resume_decision = await resume_future
#                 del pending_interruptions[client_id]

#                 if resume_decision.lower() == "y":
#                     # Resume processing using the resume command
#                     async for resumed_chunk, _ in graph.astream(Command(resume="y"), stream_mode="messages", config=config):
#                         if resumed_chunk.additional_kwargs.get("tool_calls"):
#                             tool_calls_resumed = resumed_chunk.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
#                             await websocket.send_json({"tool_calls": tool_calls_resumed})
#                         if hasattr(resumed_chunk, 'queries') and resumed_chunk.queries:
#                             await websocket.send_json({"queries": resumed_chunk.queries})
#                         if resumed_chunk.content:
#                             await websocket.send_json({"content": resumed_chunk.content})
#                         await asyncio.sleep(0.05)
#                 else:
#                     # If the frontend indicates to halt, signal the end to the UI
#                     await websocket.send_json({"end": True})
                
#                 # Stop processing after handling the interruption
#                 return

#         await websocket.send_json({"end": True})  # Completion

#     except Exception as e:
#         await websocket.send_json({"error": str(e)})


async def process_graph_stream(query: str, websocket: WebSocket, client_id: str):
    """Processes a query using the LangGraph and streams results."""
    
    input_state = InputState(messages=[query])
    thread_id = new_uuid()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        async for chunk, _ in graph.astream(input_state, stream_mode="messages", config=config):
            
            if chunk.additional_kwargs.get("tool_calls"):
                tool_calls = chunk.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
                await websocket.send_json({"tool_calls": tool_calls})

            if hasattr(chunk, 'queries') and chunk.queries:
                await websocket.send_json({"queries": chunk.queries})

            if chunk.content:
                await websocket.send_json({"content": chunk.content})

            await asyncio.sleep(0.05)


            # Handle interruptions by delegating approval to the frontend
            if hasattr(chunk, 'interrupts') and chunk.interrupts:
                prompt = chunk.interrupts["question"]
                llm_output = chunk.interrupts["llm_output"]
                binary_score = chunk.hallucination.binary_score if chunk.hallucination else "N/A"

                # Create a Future that will be resolved when the frontend sends a "resume" message
                resume_future = asyncio.Future()
                pending_interruptions[client_id] = resume_future
                
                print("Sending interrupt message:", {
                    "type": "interrupt",
                    "data": {
                        "question": prompt,
                        "llm_output": llm_output,
                        "binary_score": binary_score
                    }
                })

                # Send the interruption details to the frontend UI
                await websocket.send_json({
                    "type": "interrupt",
                    "data": {
                        "question": prompt,
                        "llm_output": llm_output,
                        "binary_score": binary_score
                    }
                })


                # Wait for the frontend to send a "resume" response (e.g., "y" or "n")
                resume_decision = await resume_future
                del pending_interruptions[client_id]

                if resume_decision.lower() == "y":
                    # Resume processing if approved by the human
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
                    # Signal the end if not approved
                    await websocket.send_json({"end": True})
                
                # Stop further processing for this chunk
                return


        await websocket.send_json({"end": True})  # Completion

    except Exception as e:
        await websocket.send_json({"error": str(e)})



# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     client_id = new_uuid()
#     await websocket.accept()
    
#     try:
#         while True:
#             data = await websocket.receive_json()
            
#             if isinstance(data, dict) and "query" in data:
#                 asyncio.create_task(process_graph_stream(data["query"], websocket, client_id))

#             # Handle interrupt response with proper type field
#             elif isinstance(data, dict) and "type" in data and data["type"] == "interrupt_response":
#                 if client_id in pending_interruptions:
#                     thread_id = pending_interruptions[client_id]["thread_id"]
#                     config = pending_interruptions[client_id]["config"]
                    
#                     del pending_interruptions[client_id]  # Remove pending state

#                     if data["response"].lower() == "y":
#                         async for resumed_chunk, _ in graph.astream(Command(resume="y"), stream_mode="messages", config=config):
#                             # Process resumed chunk
#                             if resumed_chunk.additional_kwargs.get("tool_calls"):
#                                 tool_calls_resumed = resumed_chunk.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
#                                 await websocket.send_json({"tool_calls": tool_calls_resumed})

#                             if hasattr(resumed_chunk, 'queries') and resumed_chunk.queries:
#                                 await websocket.send_json({"queries": resumed_chunk.queries})

#                             if resumed_chunk.content:
#                                 await websocket.send_json({"content": resumed_chunk.content})

#                             await asyncio.sleep(0.05)
#                     else:
#                         await websocket.send_json({"end": True})

#     except WebSocketDisconnect:
#         print("Client disconnected")
        
#         if client_id in pending_interruptions:
#             del pending_interruptions[client_id]
    
#     except Exception as e:
#         print(f"Error: {e}")



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = new_uuid()
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if isinstance(data, dict) and "query" in data:
                asyncio.create_task(process_graph_stream(data["query"], websocket, client_id))

            # Handle interrupt response with proper type field
            elif isinstance(data, dict) and "resume" in data:
                if client_id in pending_interruptions:
                    resume_future = pending_interruptions[client_id]
                    if not resume_future.done():
                        resume_future.set_result(data["resume"])

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

