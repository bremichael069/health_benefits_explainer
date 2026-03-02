"""Document loaders (notebook-aligned)."""
import os

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path: str):
    """Load PDF from path. Raises FileNotFoundError if path missing.

    The path is typically from lib.get_settings().policy_pdf_path (set via POLICY_PDF_PATH in .env).
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"PDF not found: {path}. Set POLICY_PDF_PATH or add the file under data/."
        )
    loader = PyPDFLoader(path)
    return loader.load()
