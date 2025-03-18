# subgraph/graph_builder.py

### Build Index

import os
import re
from langchain_community.vectorstores import Chroma
#from langchain_groq import GroqEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from subgraph.graph_states import ResearcherState, QueryState
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT
from langchain_core.documents import Document
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_groq import ChatGroq
from langgraph.types import Send

#Removed the Cohere Specific Imports

from rank_llm.rerank.rankllm import RankLLM
from groq_rank_llm import GroqRankLLM

#New imports
from langchain_core.runnables import chain
from operator import itemgetter

import logging
from utils.utils import config
\

load_dotenv()

logger = logging.getLogger(__name__)

# Vector store configuration
VECTORSTORE_COLLECTION = config["retriever"]["collection_name"]
VECTORSTORE_DIRECTORY = config["retriever"]["directory"]
TOP_K = config["retriever"]["top_k"]
TOP_K_COMPRESSION = config["retriever"]["top_k_compression"]
ENSEMBLE_WEIGHTS = config["retriever"]["ensemble_weights"]
#COHERE_RERANK_MODEL = config["retriever"]["cohere_rerank_model"]



def _setup_vectorstore() -> Chroma:
    """
    Set up and return the Chroma vector store instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    return Chroma(
        collection_name=VECTORSTORE_COLLECTION,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIRECTORY
    )



def _load_documents(vectorstore: Chroma) -> list[Document]:
    """
    Load documents and metadata from the vector store and return them as Langchain Document objects.

    Args:
        vectorstore (Chroma): The vector store instance.

    Returns:
        list[Document]: A list of Document objects containing the content and metadata.
    """
    all_data = vectorstore.get(include=["documents", "metadatas"])
    documents: list[Document] = []

    for content, meta in zip(all_data["documents"], all_data["metadatas"]):
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise ValueError(f"Expected metadata to be a dict, but got {type(meta)}")

        documents.append(Document(page_content=content, metadata=meta))

    return documents



def _build_retrievers(documents: list[Document], vectorstore: Chroma) -> EnsembleRetriever:
    """
    Build and return an ensemble retriever (without Cohere compression).
    """
    # Create base retriever
    retriever_bm25 = BM25Retriever.from_documents(documents, search_kwargs={"k": TOP_K})
    retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K})

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vanilla, retriever_mmr, retriever_bm25],
        weights=ENSEMBLE_WEIGHTS,
    )

    return ensemble_retriever


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate search queries based on the question (a step in the research plan).

    This function uses a language model to generate diverse search queries to help answer the question.

    Args:
        state (ResearcherState): The current state of the researcher, including the user's question.
        config (RunnableConfig): Configuration with the model used to generate queries.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing the list of generated search queries.
    """

    class Response(TypedDict):
        queries: list[str]

    logger.info("---GENERATE QUERIES---")
    model = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama3-70b-8192",max_tokens=2000, streaming=True)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": state.question},
    ]
    response = cast(Response, await model.with_structured_output(Response).ainvoke(messages))
    queries = response["queries"]
    queries.append(state.question)
    logger.info(f"Queries: {queries}")
    return {"queries": response["queries"]}


async def retrieve_and_rerank_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents and rerank them using rank_llm."""
    logger.info("---RETRIEVING DOCUMENTS---")
    logger.info(f"Query for the retrieval process: {state.query}")

    # Retrieve the initial set of documents
    documents = ensemble_retriever.get_relevant_documents(state.query)
    logger.info(f"Number of documents initially retrieved: {len(documents)}")

    #Rerank only if there are documents to rerank
    if documents:
        MAX_DOC_LENGTH = 1500
        # Prepare the documents for rank_llm
        texts = [doc.page_content[:MAX_DOC_LENGTH]  for doc in documents]

        # Define the query prompt with placeholders
        query_prompt = (
            "<s>[INST] You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, "
            "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something incorrect. "
            "If you don't know the answer to a question, please don't share false information.[/INST] "
            "Rank the documents based on the query.\n\nQuery: {query}\nDocuments: {documents} [/INST]"
        )

        # Initialize the LLMRanker
        ranker = GroqRankLLM(
            model_name="llama3-70b-8192",  # Ensure this model is available and compatible
            prompt=query_prompt,
            device="cpu",  # Adjust based on your hardware (e.g., 'cuda' for GPU)
            max_tokens=2000  # Adjust as needed
        )

        try:
            # Use the corrected rank method
            ranked_indices = ranker.rank(query=state.query, texts=texts)
            logger.info(f"Ranked indices from Groq: {ranked_indices}")

            # Create ranked_documents using the indices
            ranked_documents = [documents[i] for i in ranked_indices if 0 <= i < len(documents)]
            logger.info(f"Number of documents after reranking: {len(ranked_documents)}")


        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            ranked_documents = documents  # Fallback to original order

    else:
        ranked_documents = []

    # Log only the number of documents, not their content
    logger.info(f"Returning {len(ranked_documents)} ranked documents.")
    return {"documents": ranked_documents}


vectorstore = _setup_vectorstore()
documents = _load_documents(vectorstore)

# Build the ensemble retriever (without Cohere)
ensemble_retriever = _build_retrievers(documents, vectorstore)  # Changed to build only the EnsembleRetriever


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each generated query.

    This function prepares parallel document retrieval tasks for each query in the researcher's state.

    Args:
        state (ResearcherState): The current state of the researcher, including the generated queries.

    Returns:
        Literal["retrieve_documents"]: A list of Send objects, each representing a document retrieval task.

    Behavior:
        - Creates a Send object for each query in the state.
        - Each Send object targets the "retrieve_documents" node with the corresponding query.
    """
    return [
        Send("retrieve_and_rerank_documents", QueryState(query=query)) for query in state.queries
    ]


builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve_and_rerank_documents)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,  # type: ignore
    path_map=["retrieve_and_rerank_documents"],
)
builder.add_edge("retrieve_and_rerank_documents", END)
researcher_graph = builder.compile()
