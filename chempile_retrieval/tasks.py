from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

from .loader import load_variant_from_local_files


@dataclass(frozen=True)
class _VariantSpec:
    variant: str  # e.g. "A1"
    domain_label: str
    mode_label: str


class _ChempileRetrievalBase(AbsTaskRetrieval):
    """Local-first retrieval tasks for Chempile.

    We override load_data() to load from ./data/{variant} JSONLs.
    """

    variant_spec: _VariantSpec

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:  # noqa: ARG002
        if self.data_loaded:
            return

        repo_root = Path(__file__).resolve().parents[1]
        queries, corpus, qrels = load_variant_from_local_files(repo_root, self.variant_spec.variant)

        # MTEB expects self.dataset[subset][split] -> RetrievalSplitData
        self.dataset["default"]["test"]["queries"] = queries
        self.dataset["default"]["test"]["corpus"] = corpus
        self.dataset["default"]["test"]["relevant_docs"] = qrels
        self.dataset["default"]["test"]["top_ranked"] = None

        self.data_loaded = True


def _make_metadata(name: str, description: str) -> TaskMetadata:
    # dataset.path/revision are required by schema, even though we override load_data.
    return TaskMetadata(
        name=name,
        dataset={"path": "local", "revision": "local"},
        description=description,
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        domains=["Chemistry"],
        task_subtypes=["Question answering", "Reasoning as Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation="",
        reference=None,
    )


# --- A variants (Chemistry only)


class ChempileRetrievalA1(_ChempileRetrievalBase):
    variant_spec = _VariantSpec("A1", "chemistry", "title")
    metadata = _make_metadata(
        "ChempileRetrievalA1",
        "Chempile retrieval (chemistry only): query=title, corpus=answer (1:1).",
    )


class ChempileRetrievalA2(_ChempileRetrievalBase):
    variant_spec = _VariantSpec("A2", "chemistry", "question")
    metadata = _make_metadata(
        "ChempileRetrievalA2",
        "Chempile retrieval (chemistry only): query=question, corpus=answer (1:1).",
    )


class ChempileRetrievalA3(_ChempileRetrievalBase):
    variant_spec = _VariantSpec("A3", "chemistry", "title+question")
    metadata = _make_metadata(
        "ChempileRetrievalA3",
        "Chempile retrieval (chemistry only): query=title+question, corpus=answer (1:1).",
    )


# --- B variants (Chemistry + Matter Modeling)


class ChempileRetrievalB1(_ChempileRetrievalBase):
    variant_spec = _VariantSpec("B1", "chemistry+matter", "title")
    metadata = _make_metadata(
        "ChempileRetrievalB1",
        "Chempile retrieval (chemistry + matter): query=title, corpus=answer (1:1).",
    )


class ChempileRetrievalB2(_ChempileRetrievalBase):
    variant_spec = _VariantSpec("B2", "chemistry+matter", "question")
    metadata = _make_metadata(
        "ChempileRetrievalB2",
        "Chempile retrieval (chemistry + matter): query=question, corpus=answer (1:1).",
    )


class ChempileRetrievalB3(_ChempileRetrievalBase):
    variant_spec = _VariantSpec("B3", "chemistry+matter", "title+question")
    metadata = _make_metadata(
        "ChempileRetrievalB3",
        "Chempile retrieval (chemistry + matter): query=title+question, corpus=answer (1:1).",
    )


# --- C variants (All: Chemistry + Matter Modeling + Physics)


class ChempileRetrievalC1(_ChempileRetrievalBase):
    variant_spec = _VariantSpec("C1", "all", "title")
    metadata = _make_metadata(
        "ChempileRetrievalC1",
        "Chempile retrieval (all domains): query=title, corpus=answer (1:1).",
    )


class ChempileRetrievalC2(_ChempileRetrievalBase):
    variant_spec = _VariantSpec("C2", "all", "question")
    metadata = _make_metadata(
        "ChempileRetrievalC2",
        "Chempile retrieval (all domains): query=question, corpus=answer (1:1).",
    )


class ChempileRetrievalC3(_ChempileRetrievalBase):
    variant_spec = _VariantSpec("C3", "all", "title+question")
    metadata = _make_metadata(
        "ChempileRetrievalC3",
        "Chempile retrieval (all domains): query=title+question, corpus=answer (1:1).",
    )
