from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Send a system + user message and return the raw text response."""
