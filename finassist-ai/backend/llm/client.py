"""LLM client abstraction supporting OpenAI and Anthropic, switchable via config."""

from typing import Any

from backend.config import Settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Unified interface for calling either OpenAI or Anthropic chat models."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.llm_provider

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The fully-assembled user/context prompt.
            system: Optional system prompt override.

        Returns:
            The model's text response.
        """
        if self.provider == "openai":
            return self._complete_openai(prompt, system)
        return self._complete_anthropic(prompt, system)

    def _complete_openai(self, prompt: str, system: str | None) -> str:
        from openai import OpenAI, APIError

        client = OpenAI(api_key=self.settings.openai_api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                temperature=self.settings.llm_temperature,
                timeout=60,
            )
            return response.choices[0].message.content or ""
        except APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected OpenAI error: %s", exc)
            raise

    def _complete_anthropic(self, prompt: str, system: str | None) -> str:
        import anthropic
        from anthropic import APIError

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        try:
            response = client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=1500,
                temperature=self.settings.llm_temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected Anthropic error: %s", exc)
            raise

    def tool_calling_model(self) -> Any:
        """Return a LangChain chat model instance configured for tool calling.

        Used by the LangGraph agent, which needs a LangChain-compatible
        model object (rather than raw text completion) to bind tools to.
        """
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                temperature=self.settings.llm_temperature,
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.settings.anthropic_model,
            api_key=self.settings.anthropic_api_key,
            temperature=self.settings.llm_temperature,
        )
