class LLMUnavailableError(Exception):
    """The model provider could not complete the request."""


class LLMTimeoutError(Exception):
    """The model provider exceeded the request timeout."""


class LLMResponseError(Exception):
    """The model provider returned an invalid response."""


class RetrievalUnavailableError(Exception):
    """The document index or embedding provider is unavailable."""
