# main_graph/graph_builder.py

"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""

import asyncio
import logging
import os
import re
from typing import Any, Literal, Optional, TypedDict, Union, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_groq import ChatGroq
from langgraph.types import interrupt, Command


from subgraph.graph_builder import researcher_graph
from langchain_core.documents import Document
from typing import Any, Literal, Optional, Union

from langgraph.checkpoint.memory import MemorySaver

from utils.utils import config
from utils.summarizer import summarize_documents
from operator import itemgetter
from langchain_core.messages import BaseMessage, SystemMessage
from main_graph.graph_states import (AgentState, GradeHallucinations,
                                    InputState, Router)
from utils.prompt import (CHECK_HALLUCINATIONS, GENERAL_SYSTEM_PROMPT,
                        MORE_INFO_SYSTEM_PROMPT, RESEARCH_PLAN_SYSTEM_PROMPT,
                        RESPONSE_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT)




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.WARNING)  
logging.getLogger("urllib3").setLevel(logging.WARNING) 

logging.getLogger("openai").propagate = False
logging.getLogger("urllib3").propagate = False
logging.getLogger("httpx").propagate = False

GROQ_MODEL = "llama3-70b-8192"

async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens=2000, streaming=True)
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    ] + state.messages
    logging.info("---ANALYZE AND ROUTE QUERY---")
    logging.info(f"MESSAGES: {state.messages}")
    
    try:
        response = cast(
            Router, await model.with_structured_output(Router).ainvoke(messages)
        )
        logger.info(f"Router response: {response}")
        return {"router": response}
    except Exception as e:
        logger.error(f"Error in analyze_and_route_query: {e}")
        # Return a default Router object in case of failure
        return {"router": Router(type="general", logic=f"Error: {e}")}

    

def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    _type = itemgetter("type")(state.router) # Use itemgetter here
    logger.info(f"Routing query. Type: {_type}")  # Log the routing decision
    if _type == "environmental":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")
    

async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a environmental-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL,max_tokens=2000, streaming=True)
    messages = [
        {"role": "system", "content": RESEARCH_PLAN_SYSTEM_PROMPT}
    ] + state.messages
    logging.info("---PLAN GENERATION---")
    try:
        response = cast(Plan, await model.with_structured_output(Plan).ainvoke(messages))
        logger.info(f"Research plan: {response['steps']}")  # Log the research plan
        return {"steps": response["steps"], "documents": "delete"}
    except Exception as e:
        logger.error(f"Error in create_research_plan: {e}")
        return {"steps": [], "documents": "delete"} # Return empty plan

async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens= 2000, streaming=True)
    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                            'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    result = await researcher_graph.ainvoke({"question": state.steps[0]}) #graph call directly
    docs = result["documents"]
    step = state.steps[0]
    logging.info(f"\n{len(docs)} documents retrieved in total for the step: {step}.")
    return {"documents": result["documents"], "steps": state.steps[1:]}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    
    logger.info(f"Checking if research is finished. Remaining steps: {len(state.steps or [])}")  # Log remaining steps
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to environmental.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens=2000, streaming=True)
    system_prompt = GENERAL_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    logging.info("---RESPONSE GENERATION---")
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"

def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


async def check_hallucinations(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Analyze the user's query and checks if the response is supported by the set of facts based on the document retrieved,
    providing a binary score result.

    This function uses a language model to analyze the user's query and gives a binary score result.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens=2000, streaming=True)
    
    # Use the summarizer to reduce the document text.
    summarized_docs = summarize_documents(state.documents) if state.documents else "No documents"
    
    system_prompt = CHECK_HALLUCINATIONS.format(
        documents=summarized_docs,  # Use summarized_docs
        generation=state.messages[-1].content if state.messages else "No generation" #Get the content from messages.
    )
    
    messages = [
        {"role": "system", "content": system_prompt}
    ] + state.messages
    logging.info("---CHECK HALLUCINATIONS---")
    
    # Get the raw response as a string
    raw_response = await model.ainvoke(messages)
    logger.info(f"Raw LLM response for hallucination check: {raw_response}")

    try:
        response = cast(GradeHallucinations, await model.with_structured_output(GradeHallucinations).ainvoke(messages))
        logger.info(f"Hallucination check response: {response}")
        return {"hallucination": response}
    except Exception as e:
        logger.error(f"Error parsing hallucination check response: {e}")
        # response = None # Explicitly set to None on error
        fallback_response = GradeHallucinations(binary_score="0")  # Create fallback object LOCALLY
        return {"hallucination": fallback_response}  # Return the fallback


async def human_approval(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, str]:
    """Determines whether to retry generation based on hallucination detection."""
    logging.info("---HUMAN APPROVAL NODE---")
    logging.info(f"State Hallucination: {state.hallucination}")
    
    # When binary score is 1 (not a hallucination), proceed to END
    if state.hallucination and state.hallucination.binary_score == "1":
        return {"human_approval": "END"}
    
    # For hallucinations or failures, create an interrupt
    llm_output = state.messages[-1].content if state.messages and len(state.messages) > 0 else 'No generation to show.'
    message = "Possible hallucination detected"
    question = "Do you want to retry the generation? (y/n)"
    
    # Rather than raising an interrupt exception, store the data in the state
    state.interrupts = {
        "message": message,
        "llm_output": llm_output,
        "question": question
    }
    
    try:
        # This will be caught by the process_graph_stream function
        result = await interrupt({
            "message": message,
            "llm_output": llm_output,
            "question": question
        })
        
        if isinstance(result, str) and result.lower() == "y":
            return {"human_approval": "respond"}
        return {"human_approval": "END"}
    except Exception as e:
        logging.error(f"Interrupt failed: {e}")
        return {"human_approval": "END"}
    


# def human_approval(state: AgentState) -> bool:
#     logging.info("---HUMAN APPROVAL NODE---") # Make sure this logging is present
#     logging.info(f"State Hallucination: {state.hallucination}") # And this logging
#     if state.hallucination is None:
#         logger.error("Hallucination state is None!")
#         print(f"\nLLM Output: {state.messages[-1].content if state.messages else 'No generation to show.'}")
#         response = input("The response might not be accurate. Do you want to retry the generation? (y/n): ").strip().lower()
#         return response.lower() == 'y'  # Return True to *retry* if hallucination check failed

#     if state.hallucination.binary_score == "1":
#         return False  # Return False.  Do *NOT* interrupt.  Proceed to END.
#     else:
#         print(f"\nLLM Output: {state.messages[-1].content if state.messages else 'No generation to show.'}")
#         response = input("The response might not be accurate. Do you want to retry the generation? (y/n): ").strip().lower()
#         return response.lower() == 'y'  # Return True to *retry* if hallucination detected



# async def respond(
#     state: AgentState, *, config: RunnableConfig
# ) -> dict[str, list[BaseMessage]]:
#     """Generate a final response to the user's query based on the conducted research.

#     This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

#     Args:
#         state (AgentState): The current state of the agent, including retrieved documents and conversation history.
#         config (RunnableConfig): Configuration with the model used to respond.

#     Returns:
#         dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
#     """
#     logging.info("--- RESPONSE GENERATION STEP ---")
#     model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL,max_tokens = 2000, streaming=True)
    
#     # #Truncate each document individually before creating the context.
#     # truncated_documents = [Document(page_content = doc.page_content[:1500], metadata=doc.metadata) for doc in state.documents] #Setting the document size to 1500 characters.
#     # context = format_docs(truncated_documents)
    
#     #Truncate each document individually before creating the context.
#     summarized_docs = summarize_documents(state.documents)
#     context = format_docs(summarized_docs)
    
#     prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
#     # The `SystemMessage` type is part of the `langchain_core.messages`
#     messages = [SystemMessage(content=prompt)] + state.messages
    
#     #Removed the full_prompt creation.
#     # full_prompt = [{"role": message.role, "content": message.content} for message in messages]
#     # prompt_string = "".join(message['content'] for message in full_prompt)
#     # logging.info(f"Estimated prompt size: {len(prompt_string)} characters") #Logging the size of the prompt
    
#     response = await model.ainvoke(messages)

#     return {"messages": [response]}


# async def respond(
#     state: AgentState, *, config: RunnableConfig
# ) -> dict[str, list[BaseMessage]]:
#     logger.info("--- RESPONSE GENERATION STEP ---")
#     model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens=2000, streaming=True)

#     all_docs = state.documents
#     summarized_docs = summarize_documents(all_docs)
#     context = format_docs(summarized_docs)

#     prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
#     messages = [SystemMessage(content=prompt)] + state.messages

#     response = await model.ainvoke(messages)

#     # --- Refined Post-Processing ---
#     cleaned_response_content = response.content.strip()

#     # Remove anything before the first citation or the apology
#     match = re.search(r'(\[\d+\]|I am sorry, but I cannot answer that question)', cleaned_response_content)
#     if match:
#         cleaned_response_content = cleaned_response_content[match.start():]

#     # Remove any trailing text after the last citation (if any)
#     last_citation_match = None
#     for match in re.finditer(r'\[\d+\]', cleaned_response_content):
#         last_citation_match = match
#     if last_citation_match:
#         cleaned_response_content = cleaned_response_content[:last_citation_match.end()]

#     #Remove the based on text
#     cleaned_response_content = re.sub(r'Based on.*?(?:\.|$)', '', cleaned_response_content).strip()

#     return {"messages": [type(response)(content=cleaned_response_content, additional_kwargs=response.additional_kwargs)]}

def _extract_answer(text: str) -> str:
    """Extracts the answer and citations using regex, handling edge cases."""
    # Remove any text before the first citation or apology
    text = re.sub(r'^.*?(\[\d+\]|I am sorry, but I cannot answer that question)', r'\1', text, flags=re.DOTALL)
    # Remove any trailing text after the last citation
    text = re.sub(r'(\[\d+\])[^[]*$', r'\1', text, flags=re.DOTALL)
    text = text.strip()

    # Check if the result is just the apology message, return it directly
    if text.startswith("I am sorry, but I cannot answer that question"):
        return text

    # Find citations
    citations = re.findall(r'\[(\d+)\]', text)
    # Remove citation text, preserving order and removing duplicates
    seen_citations = set()
    unique_citations = []
    for citation in citations:
        if citation not in seen_citations:
            seen_citations.add(citation)
            unique_citations.append(citation)
    citation_string = "".join(f"[{c}]" for c in unique_citations)  # Reconstruct citation string

    # Extract the answer text, removing citation markers
    answer_text = re.sub(r'\[\d+\]', '', text).strip()

    # If there's an answer, combine it with the citations
    if answer_text:
        return f"{answer_text} {citation_string}".strip()
    return "I am sorry, but I cannot answer that question based on the provided documents."


#In main_graph/graph_builder.py
async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    logger.info("--- RESPONSE GENERATION STEP ---")
    model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens=2000, streaming=True)

    all_docs = state.documents
    summarized_docs = summarize_documents(all_docs)
    context = format_docs(summarized_docs)
    logger.info(f"Context length: {len(context)}")


    prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=prompt)] + state.messages

    try:
        # Add a timeout to the LLM call
        response = await asyncio.wait_for(model.ainvoke(messages), timeout=60.0)  # 60-second timeout
        logger.info(f"Raw LLM response: {response.content}") # Log the complete response

    except asyncio.TimeoutError:
        logger.error("Groq API call timed out!")
        return {"messages": [SystemMessage(content="I am sorry, but the request timed out. Please try again.")]}
    except Exception as e:
        logger.exception(f"Error during response generation: {e}")
        return {"messages": [SystemMessage(content=f"An unexpected error occurred: {e}")]}

    # --- Aggressive Post-Processing and Extraction ---
    cleaned_response_content = _extract_answer(response.content)
    logger.info(f"Cleaned response: {cleaned_response_content}")

    return {"messages": [type(response)(content=cleaned_response_content, additional_kwargs=response.additional_kwargs)]}

checkpointer = MemorySaver()

# builder = StateGraph(AgentState, input=InputState)
# builder.add_node("analyze_and_route_query", analyze_and_route_query)  # Use string names for nodes
# builder.add_edge(START, "analyze_and_route_query")
# builder.add_conditional_edges("analyze_and_route_query", route_query)
# builder.add_node("create_research_plan", create_research_plan)
# builder.add_node("ask_for_more_info", ask_for_more_info)
# builder.add_node("respond_to_general_query", respond_to_general_query)
# builder.add_node("conduct_research", conduct_research)
# builder.add_node("respond", respond)
# builder.add_node("check_hallucinations", check_hallucinations)

# # # Change this line:
# # builder.add_conditional_edges("check_hallucinations", human_approval, {True: "respond", False: END})

# # To this:
# builder.add_node("human_approval", human_approval)
# builder.add_edge("check_hallucinations", "human_approval")
# builder.add_conditional_edges(
#     "human_approval",
#     lambda state: state.get("human_approval", "END"),
#     {
#         "respond": "respond", 
#         "END": END
#     }
# )

# builder.add_edge("create_research_plan", "conduct_research")
# builder.add_conditional_edges("conduct_research", check_finished)

# builder.add_edge("respond", "check_hallucinations")

# graph = builder.compile(checkpointer=checkpointer)



builder = StateGraph(AgentState, input=InputState)
builder.add_node("analyze_and_route_query", analyze_and_route_query)
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_node("create_research_plan", create_research_plan)
builder.add_node("ask_for_more_info", ask_for_more_info)
builder.add_node("respond_to_general_query", respond_to_general_query)
builder.add_node("conduct_research", conduct_research)
builder.add_node("respond", respond)
builder.add_node("check_hallucinations", check_hallucinations)
builder.add_node("human_approval_node", human_approval)

# Connect the nodes correctly
builder.add_edge("check_hallucinations", "human_approval_node")
builder.add_conditional_edges(
    "human_approval_node",
    lambda state: state.human_approval if hasattr(state, "human_approval") else "END",
    {
        "respond": "respond", 
        "END": END
    }
)

builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("respond", "check_hallucinations")

graph = builder.compile(checkpointer=checkpointer)