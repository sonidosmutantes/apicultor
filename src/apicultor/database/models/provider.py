"""Provider information model."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class ProviderInfo:
    """Information about a database provider."""
    
    name: str
    display_name: str
    description: str
    is_available: bool
    is_default: bool
    supported_features: List[str]
    configuration_required: List[str]
    rate_limits: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate provider information."""
        if not self.name:
            raise ValueError("Provider name is required")
        if not self.display_name:
            raise ValueError("Provider display name is required")
    
    @property
    def supports_search(self) -> bool:
        """Check if provider supports search operations."""
        return "search" in self.supported_features
    
    @property
    def supports_download(self) -> bool:
        """Check if provider supports download operations."""
        return "download" in self.supported_features
    
    @property
    def supports_mir_search(self) -> bool:
        """Check if provider supports MIR-based search."""
        return "mir_search" in self.supported_features
    
    @property
    def supports_similar_sounds(self) -> bool:
        """Check if provider supports similar sound discovery."""
        return "similar_sounds" in self.supported_features
    
    @property
    def is_configured(self) -> bool:
        """Check if provider has all required configuration."""
        # This would be determined by the actual provider implementation
        # For now, we assume it's configured if it's available
        return self.is_available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'is_available': self.is_available,
            'is_default': self.is_default,
            'supported_features': self.supported_features,
            'configuration_required': self.configuration_required,
            'rate_limits': self.rate_limits,
            'metadata': self.metadata or {},
            'is_configured': self.is_configured,
            'capabilities': {
                'search': self.supports_search,
                'download': self.supports_download,
                'mir_search': self.supports_mir_search,
                'similar_sounds': self.supports_similar_sounds
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProviderInfo':
        """Create instance from dictionary."""
        return cls(
            name=data['name'],
            display_name=data['display_name'],
            description=data['description'],
            is_available=data['is_available'],
            is_default=data['is_default'],
            supported_features=data['supported_features'],
            configuration_required=data['configuration_required'],
            rate_limits=data.get('rate_limits'),
            metadata=data.get('metadata')
        )