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
                        RESPONSE_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT, EXTRACT_SUMMARY_PROMPT)

from utils.summarizer import summarize_documents # Ensure this is imported

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
    model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens=2000, streaming=False) # Use streaming=False for structured output

    # Use the summarizer to reduce the document text
    summarized_docs_content = format_docs(summarize_documents(state.documents)) if state.documents else "No documents provided." # Use format_docs

    # *** Use state.detailed_answer as the generation to check ***
    generation_to_check = state.detailed_answer if state.detailed_answer else "No generation available."

    # Create a more explicit prompt
    system_prompt = CHECK_HALLUCINATIONS.format(
        documents=summarized_docs_content, # Pass formatted summarized docs
        generation=generation_to_check # Pass the detailed answer
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    logging.info("---CHECK HALLUCINATIONS---")
    logging.info(f"Documents for hallucination check: {summarized_docs_content[:500]}...") # Log snippet
    logging.info(f"Generation for hallucination check: {generation_to_check}")

    try:
        # Use structured output to ensure we get a valid binary score
        response = cast(GradeHallucinations, await model.with_structured_output(GradeHallucinations).ainvoke(messages))

        # Make sure binary_score is either "1" or "0"
        if response.binary_score not in ["0", "1"]:
            logger.warning(f"Invalid binary score received: {response.binary_score}. Defaulting to '0'.")
            response.binary_score = "0" # Default to not grounded if invalid

        logger.info(f"Hallucination check response: {response}")
        return {"hallucination": response}

    except Exception as e:
        logger.error(f"Error in hallucination check: {e}")
        # Create a fallback response with binary_score="0"
        fallback_response = GradeHallucinations(binary_score="0")
        return {"hallucination": fallback_response}


def human_approval(state: AgentState) -> str | None: # Or use -> Any
    """
    Pauses the graph to wait for human approval if hallucination score is low.
    Sends interrupt data to the frontend via the backend.
    """
    logger.info("---HUMAN APPROVAL NODE---")
    logger.info(f"State Hallucination Score: {state.hallucination.binary_score if state.hallucination else 'N/A'}")

    # Proceed directly to formatting if the score is '1' (grounded)
    if state.hallucination and state.hallucination.binary_score == "1":
        logger.info("Response is grounded. Proceeding to final formatting.")
        # Return the key for the next node directly
        return "format_final_response" # Key for the edge leading to format_final_response

    # If score is '0' or hallucination state is missing, interrupt for human review
    logger.info("Response potentially not grounded or check failed. Interrupting for human approval.")

    # *** Use state.detailed_answer for the output shown to the user ***
    llm_output_to_review = state.detailed_answer if state.detailed_answer else "No detailed answer generated."

    # Prepare data for the interrupt message to the frontend
    interrupt_data = {
        "message": "Potential issue detected in the generated answer.",
        "llm_output": llm_output_to_review,
        "question": "The generated answer might not be fully accurate based on the documents. Do you want to proceed anyway, or stop?",
        "binary_score": state.hallucination.binary_score if state.hallucination else "0"
    }

    # Store interrupt data in state (optional, but can be useful for debugging)
    state.interrupts = interrupt_data

    # Use interrupt() to pause execution.
    # interrupt() doesn't return a value to the graph logic itself.
    interrupt(None)
    # Return None or omit return for the interrupt path if type hint is str | None
    return None # Explicitly return None for the interrupt path



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
    """Extracts the answer and citations, aggressively removing preamble."""
    # Remove common preamble patterns (case-insensitive, multiline)
    preamble_patterns = [
        r'^\s*Based on the provided query and documents.*?\n',
        r'^\s*Based on the provided documents.*?\n',
        r'^\s*Here is the answer to the question:?\s*\n',
        r'^\s*Ranking:?\s*\n(.*?Document \d+:.*?\n)*', # Remove ranking sections
        r'^\s*Summary: Based on the provided query.*?\n',
        r'^\s*Unfortunately, based on the provided search results.*?\n',
        r'^\s*Based on the search results.*?\n',
    ]
    cleaned_text = text.strip()
    for pattern in preamble_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Check if only an apology remains after cleaning preamble
    if re.match(r'^(I am sorry|Unfortunately|No relevant document).*?(not available|cannot answer|not provided|not contain|not mention)', cleaned_text, re.IGNORECASE):
         # Try to find the core reason if possible
         core_reason_match = re.search(r'(PUE.*?not available|CFE.*?not available|information.*?not provided)', cleaned_text, re.IGNORECASE)
         if core_reason_match:
              return core_reason_match.group(0).strip() + "." # Return just the core reason
         else:
              return "I am sorry, but I cannot answer that question based on the provided documents." # Generic fallback

    # Find citations
    citations = re.findall(r'\[(\d+)\]', cleaned_text)
    # Remove citation text, preserving order and removing duplicates
    seen_citations = set()
    unique_citations = []
    for citation in citations:
        if citation not in seen_citations:
            seen_citations.add(citation)
            unique_citations.append(citation)
    citation_string = "".join(f"[{c}]" for c in unique_citations)

    # Extract the answer text, removing citation markers and extra whitespace
    answer_text = re.sub(r'\s*\[\d+\]\s*', ' ', cleaned_text).strip()
    answer_text = re.sub(r'\s{2,}', ' ', answer_text) # Consolidate whitespace

    # Remove trailing standalone citation blocks if _extract_answer missed them initially
    answer_text = re.sub(r'\s*\[No citations available.*?\]\s*$', '', answer_text).strip()


    if answer_text:
        # Re-append unique citations if they existed
        return f"{answer_text}{' ' + citation_string if citation_string else ''}".strip()

    # If after all cleaning, no answer text remains, return apology
    return "I am sorry, but I cannot answer that question based on the provided documents."


async def extract_summary_from_answer(state: AgentState, *, config: RunnableConfig) -> dict[str, str]:
    """
    Extracts a concise summary from the generated detailed answer text.
    """
    logger.info("--- EXTRACTING SUMMARY FROM DETAILED ANSWER ---")
    # Use a capable model, maybe slightly cheaper/faster if appropriate for extraction
    model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens=500, streaming=False)

    user_query = ""
    if state.messages:
        for msg in reversed(state.messages):
            if msg.type == "human":
                user_query = msg.content
                break

    detailed_answer_text = state.detailed_answer
    if not detailed_answer_text or detailed_answer_text.startswith("Could not generate") or detailed_answer_text.startswith("I am sorry"):
        logger.warning("Detailed answer is missing or is an error message. Skipping summary extraction.")
        return {"summary": ""} # Return empty summary if detailed answer failed

    logger.info(f"Detailed answer to summarize: {detailed_answer_text[:500]}...")
    logger.info(f"Original query for summary context: {user_query}")

    prompt = EXTRACT_SUMMARY_PROMPT.format(query=user_query, detailed_answer=detailed_answer_text)
    messages = [{"role": "system", "content": prompt}]

    logger.info(f"Prompt for summary extraction: {prompt}")

    try:
        response = await model.ainvoke(messages)
        summary_text = response.content.strip()
        logger.info(f"RAW extracted summary response from LLM: {summary_text}")

        # More robust cleaning
        # 1. Remove potential LLM preamble like "Here is the summary:"
        summary_text = re.sub(r'^(SUMMARY:|Summary:|Here is the summary:)\s*', '', summary_text, flags=re.IGNORECASE).strip()

        # 2. Check if it *already* starts correctly (case-insensitive)
        required_prefix = "Based on the Google Environmental Report 2024,"
        if not summary_text.lower().startswith(required_prefix.lower()):
             logger.warning("Extracted summary didn't start as expected. Prepending required phrase.")
             summary_text = f"{required_prefix} {summary_text}"
        else:
             # Ensure the capitalization is correct if it already started correctly
             summary_text = required_prefix + summary_text[len(required_prefix):]

        logger.info(f"Cleaned extracted summary: {summary_text}")
        return {"summary": summary_text}

    except Exception as e:
        logger.error(f"Error extracting summary from detailed answer: {e}")
        return {"summary": "Could not extract summary due to an error."}


def format_docs(docs: list[Document]) -> str:
    """Convert Documents to a single string."""
    formatted = []
    for i, doc in enumerate(docs):
        # Include metadata if useful, e.g., source
        source = doc.metadata.get('source', f'Document {i+1}')
        content = doc.page_content.replace('\n', ' ').strip()
        formatted.append(f"Source: {source}\nContent: {content}")
    return "\n\n".join(formatted)

# Ensure summarize_documents is only used where appropriate (like detailed answer if needed)
# def summarize_documents... (keep this function if generate_detailed_answer still uses it)

#In main_graph/graph_builder.py
# async def respond(
#     state: AgentState, *, config: RunnableConfig
# ) -> dict[str, list[BaseMessage]]:
#     logger.info("--- RESPONSE GENERATION STEP ---")
#     model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens=2000, streaming=True)

#     all_docs = state.documents
#     summarized_docs = summarize_documents(all_docs)
#     context = format_docs(summarized_docs)
#     logger.info(f"Context length: {len(context)}")


#     prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
#     messages = [SystemMessage(content=prompt)] + state.messages

#     try:
#         response = await asyncio.wait_for(model.ainvoke(messages), timeout=60.0)
#         logger.info(f"Raw LLM response: {response.content}")

#     except asyncio.TimeoutError:
#         logger.error("Groq API call timed out!")
#         return {"messages": [SystemMessage(content="I am sorry, but the request timed out. Please try again.")]}
#     except Exception as e:
#         logger.exception(f"Error during response generation: {e}")
#         return {"messages": [SystemMessage(content=f"An unexpected error occurred: {e}")]}

#     # --- Post-processing ---
#     cleaned_response_content = _extract_answer(response.content)
#     logger.info(f"Cleaned response: {cleaned_response_content}")

#     # # Add confidence level
#     # confidence = "high" if all_docs and len(all_docs) > 2 else "moderate"
#     # logger.info(f"Inferred confidence level: {confidence}")

#     # # Remove citation references like [1], [2]
#     # cleaned_response_content = re.sub(r'\[\d+\]', '', cleaned_response_content)

#     # # Add summary at the top if the content is long
#     # # Improve summary extraction
#     # if len(cleaned_response_content) > 200:
#     #     # Get first sentence but make sure it's complete
#     #     first_period = cleaned_response_content.find('.')
#     #     if first_period > 0:
#     #         summary = cleaned_response_content[:first_period + 1].strip()
#     #     else:
#     #         # If no period found, take first 100 chars
#     #         summary = cleaned_response_content[:100].strip() + "..."
            
#     #     # Make sure the summary is meaningful
#     #     if len(summary) < 20:  # Too short to be meaningful
#     #         summary = cleaned_response_content[:100].strip() + "..."
            
#     #     final_response = f"**Key Insight:** {summary}\n\n{cleaned_response_content}"
#     # else:
#     #     final_response = cleaned_response_content

#     # In the respond function, replace the summary generation section with:

#     # --- Post-processing ---
#     cleaned_response_content = _extract_answer(response.content)
#     logger.info(f"Cleaned response: {cleaned_response_content}")

#     # Generate a semantic summary using the improved function
#     if len(cleaned_response_content) > 200:
#         summary = await generate_semantic_summary(cleaned_response_content, model)
        
#         # Add clarity for negative results
#         if "not" in cleaned_response_content.lower() and "not" not in summary.lower():
#             if re.search(r'not (available|provided|mentioned|found|present)', cleaned_response_content, re.IGNORECASE):
#                 summary = "The requested information is not available in the provided documents. " + summary
        
#         # If summary still contains ranking language, create a very simple extraction
#         if re.search(r'(rank|document|relev)', summary, re.IGNORECASE):
#             # Extract specific facts about PUE, CFE, etc.
#             key_facts = extract_key_facts(cleaned_response_content)
#             if key_facts:
#                 summary = key_facts
#             else:
#                 # Last resort - get first sentence that's not about documents/ranking
#                 sentences = re.split(r'(?<=[.!?])\s+', cleaned_response_content)
#                 for sentence in sentences:
#                     if not re.search(r'(rank|document|relev)', sentence, re.IGNORECASE):
#                         summary = sentence
#                         break
        
#         final_response = f"**Key Insight:** {summary}\n\n{cleaned_response_content}"
#     else:
#         final_response = cleaned_response_content

#     return {
#         "messages": [
#             type(response)(
#                 content=final_response,
#                 additional_kwargs=response.additional_kwargs
#             )
#         ]
#     }

# RENAME this function from 'respond' to 'generate_detailed_answer'
# async def generate_detailed_answer(
#     state: AgentState, *, config: RunnableConfig
# ) -> dict[str, str]: # Return type changed
#     logger.info("--- DETAILED ANSWER GENERATION ---") # Log message updated
#     model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens=2000, streaming=True) # Can keep streaming if needed elsewhere

#     # Context remains the same (based on summarized docs)
#     all_docs = state.documents
#     summarized_docs = summarize_documents(all_docs)
#     context = format_docs(summarized_docs)
#     logger.info(f"Context length for detailed answer: {len(context)}")

#     # Use the existing RESPONSE_SYSTEM_PROMPT, assuming it's designed for detailed answers
#     prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
#     messages = [SystemMessage(content=prompt)] + state.messages

#     try:
#         # Use invoke for non-streaming or handle streaming differently if needed later
#         response = await asyncio.wait_for(model.ainvoke(messages), timeout=60.0)
#         logger.info(f"Raw LLM detailed answer: {response.content}")

#         # Perform cleaning specific to the detailed answer
#         cleaned_detailed_answer = _extract_answer(response.content) # Use existing cleaning
#         logger.info(f"Cleaned detailed answer: {cleaned_detailed_answer}")

#         # Store the cleaned detailed answer in the state
#         return {"detailed_answer": cleaned_detailed_answer} # Store in new state field

#     except asyncio.TimeoutError:
#         logger.error("Groq API call timed out during detailed answer generation!")
#         return {"detailed_answer": "I am sorry, but the request timed out while generating the detailed answer."}
#     except Exception as e:
#         logger.exception(f"Error during detailed answer generation: {e}")
#         return {"detailed_answer": f"An unexpected error occurred while generating the detailed answer: {e}"}

# ... other imports ...
from utils.summarizer import summarize_documents # Ensure this is imported

# ... other functions ...

async def generate_detailed_answer(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, str]:
    logger.info("--- DETAILED ANSWER GENERATION ---")
    model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name=GROQ_MODEL, max_tokens=2000, streaming=True)

    all_docs = state.documents
    if not all_docs:
        logger.warning("No documents found in state for detailed answer generation.")
        return {"detailed_answer": "No relevant documents were found to answer the query."}

    # *** REVERT: Use summarize_documents again, but it now uses max_length=6000 ***
    summarized_docs_list = summarize_documents(all_docs) # This now returns a list with one doc
    if not summarized_docs_list:
         logger.warning("Summarization resulted in empty content.")
         return {"detailed_answer": "Could not generate context from documents."}

    # Use format_docs on the list containing the single summarized document
    context = format_docs(summarized_docs_list)
    logger.info(f"Context length for detailed answer (using summarized docs, max_length=6000): {len(context)}")
    # logger.debug(f"SUMMARIZED CONTEXT being sent to LLM for detailed answer:\n---\n{context}\n---") # Optional debug

    # Use the existing RESPONSE_SYSTEM_PROMPT
    prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=prompt)] + state.messages

    try:
        # Use invoke for non-streaming or handle streaming differently if needed later
        response = await asyncio.wait_for(model.ainvoke(messages), timeout=90.0) # Keep timeout
        logger.info(f"Raw LLM detailed answer: {response.content}")

        # Perform cleaning specific to the detailed answer
        cleaned_detailed_answer = _extract_answer(response.content) # Use existing cleaning
        logger.info(f"Cleaned detailed answer: {cleaned_detailed_answer}")

        # Store the cleaned detailed answer in the state
        return {"detailed_answer": cleaned_detailed_answer}

    except asyncio.TimeoutError:
        logger.error("Groq API call timed out during detailed answer generation!")
        return {"detailed_answer": "I am sorry, but the request timed out while generating the detailed answer."}
    except Exception as e:
        # Log the specific error, especially API errors
        logger.exception(f"Error during detailed answer generation: {e}")
        # Check if it's an API error and include details if possible
        error_message = f"An unexpected error occurred: {e}"
        if hasattr(e, 'message'): # Handle potential Groq API error structure
             error_message = f"API Error: {getattr(e, 'message', str(e))}"
        elif hasattr(e, 'body'): # Handle potential OpenAI/other API error structure
             error_message = f"API Error: {getattr(e, 'body', str(e))}"

        return {"detailed_answer": f"An error occurred while generating the detailed answer: {error_message}"}



# ... (keep the rest of the file, including format_docs, _extract_answer, etc.) ...

# Note: The summarize_documents function in utils/summarizer.py is now only used by check_hallucinations.
# You might consider if check_hallucinations also needs the full context or if summarized is sufficient there.
# For now, we only change generate_detailed_answer.

def format_final_response(state: AgentState) -> dict[str, list[BaseMessage]]:
    """
    Combines the generated summary and detailed answer into the final response format.
    """
    logger.info("--- FORMATTING FINAL RESPONSE ---")
    summary = state.summary
    detailed_answer = state.detailed_answer

    # Combine summary and detailed answer
    if summary and len(detailed_answer) > 0 and not detailed_answer.startswith("Could not generate") and not detailed_answer.startswith("I am sorry"):
        # Prepend summary only if detailed answer is substantial and not an error/apology
         # Check if summary is already contained within the detailed answer to avoid repetition
        if summary.lower() not in detailed_answer.lower()[:len(summary)+50]: # Check beginning
            final_content = f"**Key Finding:** {summary}\n\n{detailed_answer}"
        else:
            final_content = detailed_answer # Summary seems redundant
    else:
        # If no good summary or detailed answer, just use the detailed answer (or summary if that's all there is)
        final_content = detailed_answer or summary or "I could not generate a response."

    # Return in the expected message format
    # We need a placeholder BaseMessage type; using SystemMessage for simplicity,
    # but ideally, it should match the type expected by the graph's end state.
    return {"messages": [SystemMessage(content=final_content)]}



checkpointer = MemorySaver()

builder = StateGraph(AgentState, input=InputState)

# Add Nodes
builder.add_node("analyze_and_route_query", analyze_and_route_query)
builder.add_node("create_research_plan", create_research_plan)
builder.add_node("conduct_research", conduct_research)
# builder.add_node("generate_summary_from_docs", generate_summary_from_docs) # REMOVE THIS NODE
builder.add_node("generate_detailed_answer", generate_detailed_answer)
builder.add_node("extract_summary_from_answer", extract_summary_from_answer) # ADD THIS NEW NODE
builder.add_node("check_hallucinations", check_hallucinations)
builder.add_node("format_final_response", format_final_response)
builder.add_node("ask_for_more_info", ask_for_more_info)
builder.add_node("respond_to_general_query", respond_to_general_query)

# Define Edges
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)

# Research Path
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges(
    "conduct_research",
    check_finished,
    {
        "conduct_research": "conduct_research",
        # Change "respond" target to generate_detailed_answer
        "respond": "generate_detailed_answer"
    }
)
# builder.add_edge("generate_summary_from_docs", "generate_detailed_answer") # REMOVE THIS EDGE
builder.add_edge("generate_detailed_answer", "extract_summary_from_answer") # ADD EDGE to new node
builder.add_edge("extract_summary_from_answer", "check_hallucinations") # ADD EDGE from new node

# Hallucination Check and Final Output
builder.add_conditional_edges(
    "check_hallucinations",
    human_approval,
    {
        "format_final_response": "format_final_response",
        "y": "format_final_response"
    }
)

builder.add_edge("format_final_response", END)

# Other Paths
builder.add_edge("ask_for_more_info", END)
builder.add_edge("respond_to_general_query", END)

graph = builder.compile(checkpointer=checkpointer)
