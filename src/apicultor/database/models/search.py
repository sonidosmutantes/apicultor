"""Search request and response models."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .sound import SoundMetadata


@dataclass
class SearchRequest:
    """Search request parameters."""
    
    query: str
    limit: int = 10
    offset: int = 0
    filters: Dict[str, Any] = field(default_factory=dict)
    sort_by: Optional[str] = None
    sort_order: str = "desc"  # "asc" or "desc"
    
    def __post_init__(self):
        """Validate search parameters."""
        if not self.query:
            raise ValueError("Search query cannot be empty")
        
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        
        if self.offset < 0:
            raise ValueError("Offset cannot be negative")
        
        if self.sort_order not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")
    
    @property
    def cache_key(self) -> str:
        """Generate a cache key for this search request."""
        import hashlib
        import json
        
        # Create deterministic representation
        key_data = {
            'query': self.query,
            'limit': self.limit,
            'offset': self.offset,
            'filters': sorted(self.filters.items()) if self.filters else [],
            'sort_by': self.sort_by,
            'sort_order': self.sort_order
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    
    sounds: List[SoundMetadata]
    total_count: int
    query: str
    provider: str
    limit: int = 10
    offset: int = 0
    execution_time: Optional[float] = None  # in seconds
    cached: bool = False
    
    def __post_init__(self):
        """Validate response data."""
        if self.total_count < 0:
            raise ValueError("Total count cannot be negative")
        
        if len(self.sounds) > self.limit:
            raise ValueError("Number of sounds cannot exceed limit")
    
    @property
    def has_more_results(self) -> bool:
        """Check if there are more results available."""
        return self.offset + len(self.sounds) < self.total_count
    
    @property
    def next_offset(self) -> int:
        """Get offset for next page of results."""
        return self.offset + len(self.sounds)
    
    @property
    def page_info(self) -> Dict[str, Any]:
        """Get pagination information."""
        return {
            'current_page': (self.offset // self.limit) + 1,
            'total_pages': (self.total_count + self.limit - 1) // self.limit,
            'has_previous': self.offset > 0,
            'has_next': self.has_more_results,
            'results_in_page': len(self.sounds),
            'total_results': self.total_count
        }
    
    def get_by_id(self, sound_id: str) -> Optional[SoundMetadata]:
        """Get a specific sound from results by ID."""
        for sound in self.sounds:
            if sound.id == sound_id:
                return sound
        return None
    
    def filter_by_duration(self, min_duration: float, max_duration: float) -> 'SearchResponse':
        """Filter results by duration range."""
        filtered_sounds = [
            sound for sound in self.sounds
            if sound.duration and min_duration <= sound.duration <= max_duration
        ]
        
        return SearchResponse(
            sounds=filtered_sounds,
            total_count=len(filtered_sounds),
            query=self.query,
            provider=self.provider,
            limit=self.limit,
            offset=self.offset,
            execution_time=self.execution_time,
            cached=self.cached
        )
    
    def filter_by_tags(self, required_tags: List[str]) -> 'SearchResponse':
        """Filter results that contain all required tags."""
        required_tags_lower = [tag.lower() for tag in required_tags]
        
        filtered_sounds = []
        for sound in self.sounds:
            sound_tags_lower = [tag.lower() for tag in sound.tags]
            if all(tag in sound_tags_lower for tag in required_tags_lower):
                filtered_sounds.append(sound)
        
        return SearchResponse(
            sounds=filtered_sounds,
            total_count=len(filtered_sounds),
            query=self.query,
            provider=self.provider,
            limit=self.limit,
            offset=self.offset,
            execution_time=self.execution_time,
            cached=self.cached
        )
    
    def sort_by_quality(self, reverse: bool = True) -> 'SearchResponse':
        """Sort results by quality score."""
        sorted_sounds = sorted(
            self.sounds, 
            key=lambda s: s.quality_score, 
            reverse=reverse
        )
        
        return SearchResponse(
            sounds=sorted_sounds,
            total_count=self.total_count,
            query=self.query,
            provider=self.provider,
            limit=self.limit,
            offset=self.offset,
            execution_time=self.execution_time,
            cached=self.cached
        )
    
    def sort_by_duration(self, reverse: bool = False) -> 'SearchResponse':
        """Sort results by duration."""
        # Handle None durations by putting them at the end
        sorted_sounds = sorted(
            self.sounds,
            key=lambda s: s.duration if s.duration is not None else (float('inf') if reverse else -1),
            reverse=reverse
        )
        
        return SearchResponse(
            sounds=sorted_sounds,
            total_count=self.total_count,
            query=self.query,
            provider=self.provider,
            limit=self.limit,
            offset=self.offset,
            execution_time=self.execution_time,
            cached=self.cached
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sounds': [sound.to_dict() for sound in self.sounds],
            'total_count': self.total_count,
            'query': self.query,
            'provider': self.provider,
            'limit': self.limit,
            'offset': self.offset,
            'execution_time': self.execution_time,
            'cached': self.cached,
            'page_info': self.page_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResponse':
        """Create instance from dictionary."""
        sounds = [SoundMetadata.from_dict(sound_data) for sound_data in data['sounds']]
        
        return cls(
            sounds=sounds,
            total_count=data['total_count'],
            query=data['query'],
            provider=data['provider'],
            limit=data.get('limit', 10),
            offset=data.get('offset', 0),
            execution_time=data.get('execution_time'),
            cached=data.get('cached', False)
        )