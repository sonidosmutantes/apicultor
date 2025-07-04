"""Database plugin implementation."""

from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

from apicultor.core.interfaces import DatabaseInterface
from apicultor.database.db.FreesoundDB import FreesoundDB
from apicultor.database.db.RedPanalDB import RedPanalDB
from apicultor.database.db.JsonMirFilesData import JsonMirFilesData


logger = logging.getLogger(__name__)


class DatabasePlugin(DatabaseInterface):
    """Database plugin that provides unified access to multiple database backends."""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._providers: Dict[str, Any] = {}
        self._default_provider: Optional[str] = None
        self._enabled = False
    
    @property
    def name(self) -> str:
        """Plugin name identifier."""
        return "database"
    
    @property
    def version(self) -> str:
        """Plugin version."""
        return "1.0.0"
    
    @property
    def description(self) -> str:
        """Plugin description."""
        return "Unified database access for sound databases (Freesound, RedPanal, local files)"
    
    @property
    def dependencies(self) -> List[str]:
        """List of required plugin dependencies."""
        return []  # Database is a core plugin with no dependencies
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration.
        
        Args:
            config: Plugin-specific configuration dictionary
        """
        self._config = config
        self._default_provider = config.get("default_provider", "freesound")
        
        # Initialize available providers
        try:
            # Initialize Freesound provider
            if "freesound_api_key" in config:
                self._providers["freesound"] = FreesoundDB(config["freesound_api_key"])
                logger.info("Initialized Freesound database provider")
            
            # Initialize RedPanal provider  
            redpanal_url = config.get("redpanal_url", "http://api.redpanal.org.ar")
            self._providers["redpanal"] = RedPanalDB(redpanal_url)
            logger.info("Initialized RedPanal database provider")
            
            # Initialize local file provider
            data_dir = config.get("data_dir", "./data")
            self._providers["local"] = JsonMirFilesData(data_dir)
            logger.info("Initialized local file database provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize database providers: {e}")
            raise
        
        self._enabled = True
        logger.info(f"Database plugin initialized with {len(self._providers)} providers")
    
    def shutdown(self) -> None:
        """Clean up plugin resources."""
        for provider_name, provider in self._providers.items():
            try:
                if hasattr(provider, 'close'):
                    provider.close()
                logger.debug(f"Shutdown database provider: {provider_name}")
            except Exception as e:
                logger.warning(f"Error shutting down provider {provider_name}: {e}")
        
        self._providers.clear()
        self._enabled = False
        logger.info("Database plugin shutdown complete")
    
    @property
    def is_enabled(self) -> bool:
        """Check if plugin is currently enabled."""
        return self._enabled
    
    def search_sounds(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for sounds in the database.
        
        Args:
            query: Search query string
            filters: Optional search filters
            limit: Maximum number of results
            provider: Specific provider to use (defaults to configured default)
            
        Returns:
            List of sound metadata dictionaries
        """
        if not self._enabled:
            raise RuntimeError("Database plugin not initialized")
        
        provider_name = provider or self._default_provider
        if provider_name not in self._providers:
            raise ValueError(f"Database provider {provider_name} not available")
        
        provider_instance = self._providers[provider_name]
        
        try:
            # Call provider-specific search method
            if hasattr(provider_instance, 'search'):
                return provider_instance.search(query, filters, limit)
            elif hasattr(provider_instance, 'textSearch'):
                return provider_instance.textSearch(query, limit)
            else:
                logger.warning(f"Provider {provider_name} does not support search")
                return []
                
        except Exception as e:
            logger.error(f"Search failed for provider {provider_name}: {e}")
            return []
    
    def get_sound_by_id(
        self, 
        sound_id: str, 
        provider: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get sound metadata by ID.
        
        Args:
            sound_id: Unique sound identifier
            provider: Specific provider to use (defaults to configured default)
            
        Returns:
            Sound metadata dictionary or None if not found
        """
        if not self._enabled:
            raise RuntimeError("Database plugin not initialized")
        
        provider_name = provider or self._default_provider
        if provider_name not in self._providers:
            raise ValueError(f"Database provider {provider_name} not available")
        
        provider_instance = self._providers[provider_name]
        
        try:
            # Call provider-specific get method
            if hasattr(provider_instance, 'get_sound'):
                return provider_instance.get_sound(sound_id)
            elif hasattr(provider_instance, 'getSoundById'):
                return provider_instance.getSoundById(sound_id)
            else:
                logger.warning(f"Provider {provider_name} does not support get by ID")
                return None
                
        except Exception as e:
            logger.error(f"Get sound by ID failed for provider {provider_name}: {e}")
            return None
    
    def download_sound(
        self, 
        sound_id: str, 
        output_path: Path,
        provider: Optional[str] = None
    ) -> bool:
        """Download sound file.
        
        Args:
            sound_id: Unique sound identifier
            output_path: Path to save the downloaded file
            provider: Specific provider to use (defaults to configured default)
            
        Returns:
            True if download successful, False otherwise
        """
        if not self._enabled:
            raise RuntimeError("Database plugin not initialized")
        
        provider_name = provider or self._default_provider
        if provider_name not in self._providers:
            raise ValueError(f"Database provider {provider_name} not available")
        
        provider_instance = self._providers[provider_name]
        
        try:
            # Call provider-specific download method
            if hasattr(provider_instance, 'download'):
                return provider_instance.download(sound_id, str(output_path))
            elif hasattr(provider_instance, 'downloadSoundById'):
                return provider_instance.downloadSoundById(sound_id, str(output_path))
            else:
                logger.warning(f"Provider {provider_name} does not support download")
                return False
                
        except Exception as e:
            logger.error(f"Download failed for provider {provider_name}: {e}")
            return False
    
    def list_providers(self) -> List[str]:
        """Get list of available database providers.
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())
    
    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider information dictionary or None if not found
        """
        if provider_name not in self._providers:
            return None
        
        provider = self._providers[provider_name]
        
        return {
            "name": provider_name,
            "type": type(provider).__name__,
            "available": hasattr(provider, 'is_available') and provider.is_available(),
            "supports_search": hasattr(provider, 'search') or hasattr(provider, 'textSearch'),
            "supports_download": hasattr(provider, 'download') or hasattr(provider, 'downloadSoundById'),
        }