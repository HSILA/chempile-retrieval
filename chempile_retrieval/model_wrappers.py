from __future__ import annotations

import logging

from transformers import AutoTokenizer

from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType

logger = logging.getLogger(__name__)

# Prompts used by the Nomic family for retrieval.
# Source: ~/research/ChEmbed-Res/ChEmbedWrapper.py
_NOMIC_PROMPTS = {
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    "PairClassification": "classification: ",
    "Reranking": "classification: ",
    "STS": "classification: ",
    "Summarization": "classification: ",
    PromptType.query.value: "search_query: ",
    PromptType.document.value: "search_document: ",
}


class NomicWrapper(SentenceTransformerEncoderWrapper):
    """SentenceTransformer wrapper that injects Nomic-style prompts.

    This is required for correct behavior of Nomic-family embedding models.
    """

    def __init__(self, model_name: str, **kwargs):
        if "model_prompts" not in kwargs:
            kwargs["model_prompts"] = _NOMIC_PROMPTS
        super().__init__(model_name, **kwargs)


class ChEmbedWrapper(SentenceTransformerEncoderWrapper):
    """Wrapper for BASF-AI/ChEmbed models.

    - Injects the Nomic-style prompts.
    - For non-vanilla ChEmbed variants, replaces tokenizer with BASF-AI/ChemVocab.

    Source parity: ~/research/ChEmbed-Res/ChEmbedWrapper.py
    """

    def __init__(self, model_name: str, **kwargs):
        if "model_prompts" not in kwargs:
            kwargs["model_prompts"] = _NOMIC_PROMPTS
        super().__init__(model_name, **kwargs)

        if "BASF-AI/ChEmbed" in model_name and "vanilla" not in model_name:
            logger.info("Replacing tokenizer for %s with BASF-AI/ChemVocab", model_name)
            new_tokenizer = AutoTokenizer.from_pretrained(
                "BASF-AI/ChemVocab",
                trust_remote_code=True,
            )
            transformer_module = self.model._first_module()
            transformer_module.tokenizer = new_tokenizer
            if hasattr(transformer_module, "auto_model"):
                logger.info("New tokenizer vocab size: %s", len(new_tokenizer))
