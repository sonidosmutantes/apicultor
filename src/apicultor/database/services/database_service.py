"""Main database service coordinating multiple providers."""

from typing import List, Optional, Dict, Any, Set
from pathlib import Path
import asyncio
import logging
import time

from ..models.sound import SoundMetadata
from ..models.search import SearchRequest, SearchResponse
from ..models.provider import ProviderInfo
from ..interfaces.repository import SoundRepository
from ..exceptions import ProviderError, SoundNotFoundError, ValidationError

logger = logging.getLogger(__name__)


class DatabaseService:
    """Main database service coordinating multiple providers.
    
    This service provides a unified interface for accessing multiple
    sound database providers. It handles provider selection, error
    recovery, and result merging.
    """
    
    def __init__(
        self, 
        providers: Dict[str, SoundRepository],
        default_provider: Optional[str] = None,
        max_concurrent_requests: int = 5
    ):
        """Initialize database service.
        
        Args:
            providers: Dictionary of provider name -> repository instance
            default_provider: Name of default provider to use
            max_concurrent_requests: Maximum concurrent requests across providers
        """
        self.providers = providers
        self.default_provider_name = default_provider
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Validate default provider
        if default_provider and default_provider not in providers:
            raise ValueError(f"Default provider '{default_provider}' not found in providers")
        
        # Set default provider if not specified
        if not self.default_provider_name and providers:
            available_providers = [name for name, repo in providers.items() if repo.is_available]
            if available_providers:
                self.default_provider_name = available_providers[0]
    
    @property
    def default_provider(self) -> Optional[SoundRepository]:
        """Get the default provider repository."""
        if not self.default_provider_name:
            return None
        return self.providers.get(self.default_provider_name)
    
    def get_provider(self, name: str) -> Optional[SoundRepository]:
        """Get a specific provider by name."""
        return self.providers.get(name)
    
    def list_providers(self) -> List[ProviderInfo]:
        """List all available providers with their information."""
        provider_infos = []
        for name, repository in self.providers.items():
            try:
                info = repository.info
                # Update default status
                info = ProviderInfo(
                    name=info.name,
                    display_name=info.display_name,
                    description=info.description,
                    is_available=info.is_available,
                    is_default=(name == self.default_provider_name),
                    supported_features=info.supported_features,
                    configuration_required=info.configuration_required,
                    rate_limits=info.rate_limits,
                    metadata=info.metadata
                )
                provider_infos.append(info)
            except Exception as e:
                logger.warning(f"Failed to get info for provider {name}: {e}")
        
        return provider_infos
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [name for name, repo in self.providers.items() if repo.is_available]
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        provider: Optional[str] = None,
        merge_providers: bool = False,
        **filters
    ) -> SearchResponse:
        """Search for sounds across providers.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            offset: Result offset for pagination
            provider: Specific provider to search (None for default)
            merge_providers: Whether to search all providers and merge results
            **filters: Additional search filters
            
        Returns:
            Search response with results
            
        Raises:
            ValidationError: If search parameters are invalid
            ProviderError: If specified provider is not available
        """
        start_time = time.time()
        
        # Create search request
        request = SearchRequest(
            query=query,
            limit=limit,
            offset=offset,
            filters=filters
        )
        
        if provider:
            # Search specific provider
            if provider not in self.providers:
                raise ProviderError(provider, "Provider not found")
            if not self.providers[provider].is_available:
                raise ProviderError(provider, "Provider not available")
            
            response = await self._search_provider(self.providers[provider], request)
            
        elif merge_providers:
            # Search all providers and merge results
            response = await self._search_all_providers(request)
            
        else:
            # Search default provider
            if not self.default_provider:
                raise ProviderError("none", "No default provider available")
            
            response = await self._search_provider(self.default_provider, request)
        
        # Add execution time
        execution_time = time.time() - start_time
        return SearchResponse(
            sounds=response.sounds,
            total_count=response.total_count,
            query=response.query,
            provider=response.provider,
            limit=response.limit,
            offset=response.offset,
            execution_time=execution_time,
            cached=response.cached
        )
    
    async def _search_provider(
        self, 
        repository: SoundRepository, 
        request: SearchRequest
    ) -> SearchResponse:
        """Search a single provider."""
        async with self._semaphore:
            try:
                return await repository.search(request)
            except Exception as e:
                logger.error(f"Search failed for provider {repository.name}: {e}")
                raise ProviderError(repository.name, f"Search failed: {str(e)}")
    
    async def _search_all_providers(self, request: SearchRequest) -> SearchResponse:
        """Search all available providers and merge results."""
        available_providers = [
            (name, repo) for name, repo in self.providers.items() 
            if repo.is_available
        ]
        
        if not available_providers:
            return SearchResponse(
                sounds=[],
                total_count=0,
                query=request.query,
                provider="none"
            )
        
        # Create tasks for all providers
        tasks = [
            self._safe_search_provider(name, repo, request)
            for name, repo in available_providers
        ]
        
        # Execute searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge successful results
        all_sounds = []
        total_count = 0
        provider_names = []
        
        for i, result in enumerate(results):
            if isinstance(result, SearchResponse):
                all_sounds.extend(result.sounds)
                total_count += result.total_count
                provider_names.append(available_providers[i][0])
            elif isinstance(result, Exception):
                logger.warning(f"Provider {available_providers[i][0]} search failed: {result}")
        
        # Remove duplicates (same sound from different providers)
        unique_sounds = self._deduplicate_sounds(all_sounds)
        
        # Sort by quality and limit results
        sorted_sounds = sorted(unique_sounds, key=lambda s: s.quality_score, reverse=True)
        limited_sounds = sorted_sounds[request.offset:request.offset + request.limit]
        
        return SearchResponse(
            sounds=limited_sounds,
            total_count=total_count,
            query=request.query,
            provider=",".join(provider_names) if provider_names else "none",
            limit=request.limit,
            offset=request.offset
        )
    
    async def _safe_search_provider(
        self, 
        name: str, 
        repository: SoundRepository, 
        request: SearchRequest
    ) -> Optional[SearchResponse]:
        """Safely search a provider, returning None on error."""
        try:
            return await self._search_provider(repository, request)
        except Exception as e:
            logger.warning(f"Provider {name} search failed: {e}")
            return None
    
    def _deduplicate_sounds(self, sounds: List[SoundMetadata]) -> List[SoundMetadata]:
        """Remove duplicate sounds based on name and duration similarity."""
        unique_sounds = []
        seen_signatures = set()
        
        for sound in sounds:
            # Create signature based on name and duration
            name_normalized = sound.name.lower().strip()
            duration_rounded = round(sound.duration, 1) if sound.duration else None
            signature = (name_normalized, duration_rounded)
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_sounds.append(sound)
        
        return unique_sounds
    
    async def get_by_id(
        self,
        sound_id: str,
        provider: Optional[str] = None
    ) -> Optional[SoundMetadata]:
        """Get sound metadata by ID.
        
        Args:
            sound_id: Sound identifier
            provider: Specific provider to search (None to search all)
            
        Returns:
            Sound metadata if found, None otherwise
        """
        if not sound_id:
            raise ValidationError("Sound ID cannot be empty")
        
        if provider:
            # Search specific provider
            if provider not in self.providers:
                raise ProviderError(provider, "Provider not found")
            
            return await self.providers[provider].get_by_id(sound_id)
        
        # Try default provider first
        if self.default_provider:
            try:
                metadata = await self.default_provider.get_by_id(sound_id)
                if metadata:
                    return metadata
            except SoundNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Default provider {self.default_provider_name} failed: {e}")
        
        # Try other providers
        for name, repository in self.providers.items():
            if name == self.default_provider_name or not repository.is_available:
                continue
            
            try:
                metadata = await repository.get_by_id(sound_id)
                if metadata:
                    return metadata
            except SoundNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Provider {name} get_by_id failed: {e}")
                continue
        
        return None
    
    async def download(
        self,
        sound_id: str,
        output_path: Path,
        provider: Optional[str] = None
    ) -> Optional[Path]:
        """Download sound file.
        
        Args:
            sound_id: Sound identifier
            output_path: Where to save the file
            provider: Specific provider to use (None for auto-detection)
            
        Returns:
            Path where file was saved, None if failed
        """
        if not sound_id:
            raise ValidationError("Sound ID cannot be empty")
        
        # First, find which provider has this sound
        metadata = await self.get_by_id(sound_id, provider)
        if not metadata:
            raise SoundNotFoundError(sound_id, provider or "any")
        
        # Use the provider that found the sound
        target_provider = provider or metadata.provider
        if target_provider not in self.providers:
            raise ProviderError(target_provider, "Provider not available")
        
        try:
            return await self.providers[target_provider].download(sound_id, output_path)
        except Exception as e:
            logger.error(f"Download failed for {sound_id} from {target_provider}: {e}")
            raise ProviderError(target_provider, f"Download failed: {str(e)}")
    
    async def get_similar(
        self,
        sound_id: str,
        limit: int = 10,
        provider: Optional[str] = None
    ) -> List[SoundMetadata]:
        """Get sounds similar to the given sound.
        
        Args:
            sound_id: Reference sound ID
            limit: Maximum number of similar sounds
            provider: Specific provider to use
            
        Returns:
            List of similar sounds
        """
        if provider:
            # Use specific provider
            if provider not in self.providers:
                raise ProviderError(provider, "Provider not found")
            
            return await self.providers[provider].get_similar(sound_id, limit)
        
        # Find the provider that has this sound
        metadata = await self.get_by_id(sound_id)
        if not metadata:
            raise SoundNotFoundError(sound_id, "any")
        
        target_provider = metadata.provider
        if target_provider not in self.providers:
            raise ProviderError(target_provider, "Provider not available")
        
        return await self.providers[target_provider].get_similar(sound_id, limit)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all providers.
        
        Returns:
            Dictionary with health status for each provider
        """
        health_results = {}
        
        tasks = [
            self._check_provider_health(name, repo)
            for name, repo in self.providers.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (name, _) in enumerate(self.providers.items()):
            if isinstance(results[i], dict):
                health_results[name] = results[i]
            else:
                health_results[name] = {
                    "status": "error",
                    "error": str(results[i]),
                    "available": False
                }
        
        # Overall health
        healthy_count = sum(1 for result in health_results.values() 
                          if result.get("status") == "healthy")
        total_count = len(health_results)
        
        return {
            "overall_status": "healthy" if healthy_count > 0 else "unhealthy",
            "healthy_providers": healthy_count,
            "total_providers": total_count,
            "default_provider": self.default_provider_name,
            "providers": health_results
        }
    
    async def _check_provider_health(self, name: str, repository: SoundRepository) -> Dict[str, Any]:
        """Check health of a single provider."""
        try:
            health_info = await repository.health_check()
            health_info["available"] = repository.is_available
            return health_info
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "available": False
            }
    
    async def close(self):
        """Close all provider connections."""
        close_tasks = [
            repo.close() for repo in self.providers.values()
        ]
        await asyncio.gather(*close_tasks, return_exceptions=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()