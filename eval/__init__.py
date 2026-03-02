"""Eval package: RAGAS evaluation and LangSmith. Run with: python -m eval.eval."""
# Lazy imports to avoid loading heavy deps on import
def get_eval_dataset():
    from eval.eval import get_eval_dataset as _f
    return _f()

def get_retriever_names():
    from eval.eval import get_retriever_names as _f
    return _f()

__all__ = ["get_eval_dataset", "get_retriever_names"]
