import os
import re


class RAGGenerator:
    """Answers strictly from the retrieved passages and cites them as [1], [2], ..."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if self.api_key:
            # Groq exposes an OpenAI-compatible endpoint, so the OpenAI client works unchanged.
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
        else:
            self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @staticmethod
    def cited_sources(answer: str, n_contexts: int) -> list[int]:
        """1-based passage numbers the answer actually cites, in order of first use."""
        seen: list[int] = []
        for match in re.findall(r"\[(\d+)\]", answer or ""):
            n = int(match)
            if 1 <= n <= n_contexts and n not in seen:
                seen.append(n)
        return seen

    def generate_response(self, query: str, contexts: list[str]) -> str:
        if not self.client:
            return "Answer generation is off. Set GROQ_API_KEY on the backend to enable it."

        context_block = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(contexts))
        system_prompt = (
            "Answer the question using ONLY the numbered passages. After every claim, cite the passage "
            "it came from as [n]. If the passages do not contain the answer, reply exactly: "
            "\"The indexed documents do not answer this.\" Never use outside knowledge."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Passages:\n{context_block}\n\nQuestion: {query}"},
                ],
                temperature=0.2,
                max_tokens=350,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Answer generation failed: {e}"
