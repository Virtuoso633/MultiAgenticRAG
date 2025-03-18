# utils/summarizer.py

from langchain_core.documents import Document  # Import Document

def truncate_text(text: str, max_length: int = 1500) -> str:
    """
    Truncate text to a maximum character length.
    You might want to adjust this based on token estimates.
    """
    return text if len(text) <= max_length else text[:max_length] + "..."


def summarize_documents(docs: list, max_length: int = 1500) -> list[Document]:
    """
    Given a list of Document objects, concatenate their content, then truncate the result.
    Returns a list containing a single Document object with the summarized content.
    """
    if not docs:
        return []  # Handle empty document list

    combined_text = ""
    for doc in docs:
        if doc.page_content:  # Only add if the document has content
            combined_text += doc.page_content + "\n"  # Add a newline between documents for readability

    if not combined_text:
        return [] #Handle empty combined text

    #Truncate the combined text using a maximum length
    truncated_text = truncate_text(combined_text, max_length=max_length)
    return [Document(page_content=truncated_text)]  # Wrap in Document object

