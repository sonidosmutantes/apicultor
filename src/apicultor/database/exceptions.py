"""Database-specific exceptions."""

from typing import Optional


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ProviderError(DatabaseError):
    """Provider-specific error."""
    
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class ConnectionError(ProviderError):
    """Connection-related error."""
    pass


class AuthenticationError(ProviderError):
    """Authentication error."""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded error."""
    
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after} seconds"
        super().__init__(provider, message)


class ValidationError(DatabaseError):
    """Input validation error."""
    pass


class SoundNotFoundError(DatabaseError):
    """Sound not found error."""
    
    def __init__(self, sound_id: str, provider: str):
        self.sound_id = sound_id
        self.provider = provider
        super().__init__(f"Sound '{sound_id}' not found in provider '{provider}'")