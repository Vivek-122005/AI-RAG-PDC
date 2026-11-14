"""Answer pipelines: the baseline now, the self-correcting agent in Phase 3."""

from agentic_rag.pipelines.answer import Answer, is_refusal
from agentic_rag.pipelines.baseline import BaselineRAG

__all__ = ["Answer", "BaselineRAG", "is_refusal"]
