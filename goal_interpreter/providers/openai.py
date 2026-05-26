from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str | None = None):
        """Works with OpenAI, Azure OpenAI, or any OpenAI-compatible endpoint (e.g. Ollama)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install the openai package: pip install openai")
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=3600)
        self._model = model

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=4096,
            extra_body={"options": {"num_ctx": 8192, "num_predict": 4096}},
        )
        return response.choices[0].message.content
