"""Text splitters (notebook-aligned)."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Main chunks for vector store (Task 2, Task 4)
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# Parent Document Retriever (Task 8)
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50


def split_docs(docs, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    """Split documents with RecursiveCharacterTextSplitter (notebook: health_benefits_docs)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def get_parent_splitter():
    """RecursiveCharacterTextSplitter for parent docs (Task 8). chunk_size=2000, chunk_overlap=200."""
    return RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )


def get_child_splitter():
    """RecursiveCharacterTextSplitter for child chunks (Task 8). chunk_size=400, chunk_overlap=50."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )
