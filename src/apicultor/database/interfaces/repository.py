"""Abstract repository interface for sound database operations."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterable
from pathlib import Path

from ..models.sound import SoundMetadata
from ..models.search import SearchRequest, SearchResponse
from ..models.provider import ProviderInfo


class SoundRepository(ABC):
    """Abstract repository for sound database operations.
    
    This interface defines the contract that all database providers must implement.
    It provides a consistent API for searching, retrieving, and downloading sounds
    regardless of the underlying data source.
    """
    
    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Search for sounds based on criteria.
        
        Args:
            request: Search request with query, filters, and pagination
            
        Returns:
            Search response with matching sounds and metadata
            
        Raises:
            ValidationError: If search parameters are invalid
            ProviderError: If provider-specific error occurs
            ConnectionError: If connection to provider fails
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, sound_id: str) -> Optional[SoundMetadata]:
        """Get sound metadata by ID.
        
        Args:
            sound_id: Unique identifier for the sound
            
        Returns:
            Sound metadata if found, None otherwise
            
        Raises:
            ValidationError: If sound_id is invalid
            ProviderError: If provider-specific error occurs
            SoundNotFoundError: If sound does not exist
        """
        pass
    
    @abstractmethod
    async def download(self, sound_id: str, output_path: Path) -> Path:
        """Download sound file to specified path.
        
        Args:
            sound_id: Unique identifier for the sound
            output_path: Path where file should be saved
            
        Returns:
            Actual path where file was saved
            
        Raises:
            ValidationError: If parameters are invalid
            SoundNotFoundError: If sound does not exist
            PermissionError: If cannot write to output path
            ConnectionError: If download fails
        """
        pass
    
    @abstractmethod
    async def get_similar(self, sound_id: str, limit: int = 10) -> List[SoundMetadata]:
        """Get sounds similar to the given sound.
        
        Args:
            sound_id: Reference sound ID
            limit: Maximum number of similar sounds to return
            
        Returns:
            List of similar sounds, ordered by similarity
            
        Raises:
            ValidationError: If parameters are invalid
            SoundNotFoundError: If reference sound does not exist
            ProviderError: If similarity search is not supported
        """
        pass
    
    @abstractmethod
    async def search_by_mir_features(
        self, 
        features: Dict[str, Any], 
        limit: int = 10,
        tolerance: float = 0.1
    ) -> List[SoundMetadata]:
        """Search sounds by MIR features.
        
        Args:
            features: Dictionary of MIR feature names and target values
            limit: Maximum number of results to return
            tolerance: Tolerance for feature matching (0.0 to 1.0)
            
        Returns:
            List of sounds with matching features
            
        Raises:
            ValidationError: If features or parameters are invalid
            ProviderError: If MIR search is not supported
        """
        pass
    
    @abstractmethod
    async def get_random_sounds(self, count: int = 10, **filters) -> List[SoundMetadata]:
        """Get random sounds from the database.
        
        Args:
            count: Number of random sounds to return
            **filters: Optional filters to apply (tags, duration, etc.)
            
        Returns:
            List of random sounds matching filters
            
        Raises:
            ValidationError: If parameters are invalid
            ProviderError: If random selection is not supported
        """
        pass
    
    # Provider information and capabilities
    
    @property
    @abstractmethod
    def info(self) -> ProviderInfo:
        """Get provider information and capabilities."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get provider name."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health and connectivity.
        
        Returns:
            Dictionary with health status information
        """
        pass
    
    # Statistics and metadata
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get provider statistics.
        
        Returns:
            Dictionary with statistics like total sounds, last update, etc.
        """
        pass
    
    @abstractmethod
    async def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats.
        
        Returns:
            List of file extensions (e.g., ['.wav', '.mp3', '.ogg'])
        """
        pass
    
    @abstractmethod
    async def get_supported_mir_features(self) -> List[str]:
        """Get list of supported MIR features.
        
        Returns:
            List of MIR feature names available for search
        """
        pass
    
    # Batch operations
    
    async def get_multiple_by_ids(self, sound_ids: List[str]) -> List[Optional[SoundMetadata]]:
        """Get multiple sounds by their IDs.
        
        Default implementation calls get_by_id for each ID.
        Providers can override for batch optimization.
        
        Args:
            sound_ids: List of sound IDs to retrieve
            
        Returns:
            List of sound metadata (None for not found sounds)
        """
        results = []
        for sound_id in sound_ids:
            try:
                metadata = await self.get_by_id(sound_id)
                results.append(metadata)
            except Exception:
                results.append(None)
        return results
    
    async def download_multiple(
        self, 
        sound_ids: List[str], 
        output_dir: Path
    ) -> List[Optional[Path]]:
        """Download multiple sounds to a directory.
        
        Default implementation calls download for each ID.
        Providers can override for batch optimization.
        
        Args:
            sound_ids: List of sound IDs to download
            output_dir: Directory where files should be saved
            
        Returns:
            List of file paths (None for failed downloads)
        """
        results = []
        for sound_id in sound_ids:
            try:
                output_path = output_dir / f"{sound_id}"
                file_path = await self.download(sound_id, output_path)
                results.append(file_path)
            except Exception:
                results.append(None)
        return results
    
    # Context manager support
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    @abstractmethod
    async def close(self):
        """Close connections and clean up resources."""
        pass