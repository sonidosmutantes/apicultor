"""Sound metadata domain model."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class SoundMetadata:
    """Immutable sound metadata model.
    
    This represents sound metadata from any provider in a normalized format.
    All fields are immutable to ensure data integrity.
    """
    # Required fields
    id: str
    name: str
    provider: str
    
    # Optional descriptive fields
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    license: Optional[str] = None
    created_at: Optional[datetime] = None
    
    # Audio technical properties
    duration: Optional[float] = None  # in seconds
    sample_rate: Optional[int] = None  # in Hz
    channels: Optional[int] = None
    file_format: Optional[str] = None
    file_size: Optional[int] = None  # in bytes
    
    # URLs and access
    url: Optional[str] = None  # Main URL on provider site
    download_url: Optional[str] = None  # Direct download URL
    preview_url: Optional[str] = None  # Preview/streaming URL
    
    # MIR features and analysis
    mir_features: Optional[Dict[str, Any]] = None
    
    # Provider-specific metadata
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.id:
            raise ValueError("Sound ID is required")
        if not self.provider:
            raise ValueError("Provider is required")
        if not self.name:
            raise ValueError("Sound name is required")
    
    @property
    def duration_formatted(self) -> str:
        """Get human-readable duration string."""
        if not self.duration:
            return "Unknown"
        
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    @property
    def file_size_formatted(self) -> str:
        """Get human-readable file size string."""
        if not self.file_size:
            return "Unknown"
        
        # Convert bytes to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.file_size < 1024.0:
                return f"{self.file_size:.1f} {unit}"
            self.file_size /= 1024.0
        return f"{self.file_size:.1f} TB"
    
    @property
    def quality_score(self) -> float:
        """Calculate a quality score for this sound (0-100)."""
        score = 0.0
        
        # Duration scoring (prefer 5-30 seconds)
        if self.duration:
            if 5 <= self.duration <= 30:
                score += 25
            elif 1 <= self.duration <= 60:
                score += 15
            elif self.duration > 0:
                score += 5
        
        # Sample rate scoring
        if self.sample_rate:
            if self.sample_rate >= 44100:
                score += 20
            elif self.sample_rate >= 22050:
                score += 10
            else:
                score += 5
        
        # Format scoring (prefer lossless formats)
        if self.file_format:
            format_lower = self.file_format.lower()
            if format_lower in ['wav', 'flac', 'aiff']:
                score += 15
            elif format_lower in ['mp3', 'ogg', 'm4a']:
                score += 10
            else:
                score += 5
        
        # Description and tags scoring
        if self.description and len(self.description) > 10:
            score += 10
        if self.tags and len(self.tags) > 0:
            score += min(len(self.tags) * 2, 10)
        
        # License scoring (prefer open licenses)
        if self.license:
            license_lower = self.license.lower()
            if any(term in license_lower for term in ['cc', 'creative commons', 'public domain']):
                score += 10
            else:
                score += 5
        
        # Provider-specific scoring
        if self.provider == "freesound":
            num_ratings = self.extra_metadata.get('num_ratings', 0)
            avg_rating = self.extra_metadata.get('avg_rating', 0)
            if num_ratings > 0 and avg_rating > 0:
                # Freesound ratings are 0-5, normalize to 0-10
                rating_score = (avg_rating / 5.0) * 10
                # Weight by number of ratings (more ratings = more reliable)
                weight = min(num_ratings / 10.0, 1.0)
                score += rating_score * weight
        
        return min(score, 100.0)  # Cap at 100
    
    def has_mir_feature(self, feature_name: str) -> bool:
        """Check if a specific MIR feature is available."""
        if not self.mir_features:
            return False
        
        # Support nested feature names like "lowlevel.spectral_centroid"
        feature_parts = feature_name.split('.')
        current_level = self.mir_features
        
        for part in feature_parts:
            if not isinstance(current_level, dict) or part not in current_level:
                return False
            current_level = current_level[part]
        
        return True
    
    def get_mir_feature(self, feature_name: str, default: Any = None) -> Any:
        """Get a specific MIR feature value."""
        if not self.mir_features:
            return default
        
        # Support nested feature names
        feature_parts = feature_name.split('.')
        current_level = self.mir_features
        
        for part in feature_parts:
            if not isinstance(current_level, dict) or part not in current_level:
                return default
            current_level = current_level[part]
        
        return current_level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'provider': self.provider,
            'description': self.description,
            'tags': self.tags,
            'license': self.license,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'file_format': self.file_format,
            'file_size': self.file_size,
            'url': self.url,
            'download_url': self.download_url,
            'preview_url': self.preview_url,
            'mir_features': self.mir_features,
            'extra_metadata': self.extra_metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SoundMetadata':
        """Create instance from dictionary."""
        # Handle datetime parsing
        created_at = None
        if data.get('created_at'):
            if isinstance(data['created_at'], str):
                created_at = datetime.fromisoformat(data['created_at'])
            elif isinstance(data['created_at'], datetime):
                created_at = data['created_at']
        
        return cls(
            id=data['id'],
            name=data['name'],
            provider=data['provider'],
            description=data.get('description'),
            tags=data.get('tags', []),
            license=data.get('license'),
            created_at=created_at,
            duration=data.get('duration'),
            sample_rate=data.get('sample_rate'),
            channels=data.get('channels'),
            file_format=data.get('file_format'),
            file_size=data.get('file_size'),
            url=data.get('url'),
            download_url=data.get('download_url'),
            preview_url=data.get('preview_url'),
            mir_features=data.get('mir_features'),
            extra_metadata=data.get('extra_metadata', {}),
        )