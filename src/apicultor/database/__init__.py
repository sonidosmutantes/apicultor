"""Modern database module for Apicultor.

This module provides a clean, async-first database abstraction layer
for accessing sound databases like Freesound, RedPanal, and local files.
"""

from .services.database_service import DatabaseService
from .models.sound import SoundMetadata
from .models.search import SearchRequest, SearchResponse
from .exceptions import (
    DatabaseError,
    ProviderError, 
    ConnectionError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    SoundNotFoundError
)

__all__ = [
    "DatabaseService",
    "SoundMetadata",
    "SearchRequest", 
    "SearchResponse",
    "DatabaseError",
    "ProviderError",
    "ConnectionError", 
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "SoundNotFoundError",
]