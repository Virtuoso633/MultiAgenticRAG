# utils/summarizer.py

import logging

logger = logging.getLogger(__name__)
from langchain_core.documents import Document  # Import Document

def truncate_text(text: str, max_length: int = 1500) -> str:
    """
    Truncate text to a maximum character length.
    You might want to adjust this based on token estimates.
    """
    return text if len(text) <= max_length else text[:max_length] + "..."


# Increase the default max_length significantly
def summarize_documents(docs: list[Document], max_length: int = 6000) -> list[Document]:
    """
    Concatenates page content of documents and truncates to max_length.
    Returns a single Document with the summarized content.
    """
    logger.info(f"Summarizing {len(docs)} documents with max_length={max_length}")
    if not docs:
        return []

    # Concatenate content from all documents
    full_content = "\n\n".join([doc.page_content for doc in docs])

    # Truncate if necessary
    if len(full_content) > max_length:
        truncated_content = full_content[:max_length]
        logger.warning(f"Combined document content length ({len(full_content)}) exceeded max_length ({max_length}). Truncated.")
    else:
        truncated_content = full_content
        logger.info(f"Combined document content length: {len(truncated_content)}")


    # Return as a single Document (or adjust if downstream expects list)
    # Assuming downstream like format_docs can handle a list with one item
    summary_doc = Document(page_content=truncated_content, metadata={"source": "summarized_context"})
    return [summary_doc] # Return as a list containing one document