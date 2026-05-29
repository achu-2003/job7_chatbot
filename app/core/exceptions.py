class ChatbotError(Exception):
    """Base class for application errors."""


class UnsafeSQLError(ChatbotError):
    """Raised when an AI-generated SQL fragment fails safety checks."""


class HallucinationError(ChatbotError):
    """Raised when the LLM response cannot be grounded in retrieved context."""


class EscalationRequired(ChatbotError):
    """Raised when the orchestrator decides the conversation must go to a human."""
