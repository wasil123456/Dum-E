"""Local conversation summarization for DUM-E."""

from __future__ import annotations

from dume.models import Turn


class SumySummarizer:
    """Summarize turn answers locally with Sumy's LexRank algorithm."""

    def __init__(self, sentences_count: int = 2) -> None:
        """Create a summarizer producing the requested number of sentences."""
        if sentences_count <= 0:
            raise ValueError(
                f"sentences_count must be positive, got {sentences_count}"
            )
        self.sentences_count = sentences_count

    def summarize(self, turn: Turn | str) -> str:
        """Return a local extractive summary of a turn answer or text.

        Sumy and NLTK are imported lazily so constructing the memory manager
        does not load NLP dependencies or require a tokenizer until needed.
        """
        text = turn.answer if isinstance(turn, Turn) else turn
        cleaned = text.strip()
        if not cleaned:
            return ""

        try:
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.summarizers.lex_rank import LexRankSummarizer
        except ImportError as exc:
            raise RuntimeError(
                "Sumy and NLTK are required for local conversation summarization."
            ) from exc

        try:
            parser = PlaintextParser.from_string(cleaned, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            sentences = summarizer(parser.document, self.sentences_count)
        except LookupError as exc:
            raise RuntimeError(
                "The NLTK English tokenizer data is unavailable. "
                "Install the required NLTK tokenizer data before summarizing."
            ) from exc

        return " ".join(str(sentence).strip() for sentence in sentences).strip()
