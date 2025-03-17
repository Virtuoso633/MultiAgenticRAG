# utils/summarizer.py

def truncate_text(text: str, max_length: int = 2000) -> str:
    """
    Truncate text to a maximum character length.
    You might want to adjust this based on token estimates.
    """
    return text if len(text) <= max_length else text[:max_length] + "..."

def summarize_documents(docs: list) -> str:
    """
    Given a list of Document objects, concatenate their content,
    then truncate the result. In a more advanced implementation, you
    could run a summarization chain.
    """
    # For simplicity, concatenate all document contents.
    combined = "\n".join(doc.page_content for doc in docs)
    # Adjust max_length as needed (here 2000 characters is used as an example)
    return truncate_text(combined, max_length=2000)
