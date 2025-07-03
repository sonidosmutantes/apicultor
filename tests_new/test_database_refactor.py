"""Test cases for the refactored database module."""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from apicultor.database import (
    DatabaseService,
    SoundMetadata,
    SearchRequest,
    SearchResponse,
    DatabaseError,
    ProviderError,
    SoundNotFoundError,
    ValidationError
)
from apicultor.database.interfaces.repository import SoundRepository
from apicultor.database.models.provider import ProviderInfo


class MockRepository(SoundRepository):
    """Mock repository for testing."""
    
    def __init__(self, name: str, available: bool = True):
        self._name = name
        self._available = available
        self._sounds = {}  # sound_id -> SoundMetadata
        self._search_results = {}  # query -> list of sounds
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self._name,
            display_name=f"{self._name.title()} Provider",
            description=f"Mock {self._name} provider for testing",
            is_available=self._available,
            is_default=False,
            supported_features=["search", "download", "mir_search"],
            configuration_required=[]
        )
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Mock search implementation."""
        sounds = self._search_results.get(request.query, [])
        limited_sounds = sounds[request.offset:request.offset + request.limit]
        
        return SearchResponse(
            sounds=limited_sounds,
            total_count=len(sounds),
            query=request.query,
            provider=self._name,
            limit=request.limit,
            offset=request.offset
        )
    
    async def get_by_id(self, sound_id: str) -> SoundMetadata:
        """Mock get by ID implementation."""
        if sound_id not in self._sounds:
            raise SoundNotFoundError(sound_id, self._name)
        return self._sounds[sound_id]
    
    async def download(self, sound_id: str, output_path: Path) -> Path:
        """Mock download implementation."""
        if sound_id not in self._sounds:
            raise SoundNotFoundError(sound_id, self._name)
        # Simulate file creation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("mock audio data")
        return output_path
    
    async def get_similar(self, sound_id: str, limit: int = 10) -> list:
        """Mock similar sounds implementation."""
        if sound_id not in self._sounds:
            raise SoundNotFoundError(sound_id, self._name)
        # Return empty list for mock
        return []
    
    async def search_by_mir_features(self, features: dict, limit: int = 10, tolerance: float = 0.1) -> list:
        """Mock MIR search implementation."""
        return []
    
    async def get_random_sounds(self, count: int = 10, **filters) -> list:
        """Mock random sounds implementation."""
        return []
    
    async def health_check(self) -> dict:
        """Mock health check implementation."""
        return {
            "status": "healthy" if self._available else "unhealthy",
            "available": self._available
        }
    
    async def get_statistics(self) -> dict:
        """Mock statistics implementation."""
        return {
            "total_sounds": len(self._sounds),
            "provider": self._name
        }
    
    async def get_supported_formats(self) -> list:
        """Mock supported formats implementation."""
        return [".wav", ".mp3", ".ogg"]
    
    async def get_supported_mir_features(self) -> list:
        """Mock supported MIR features implementation."""
        return ["spectral_centroid", "tempo", "loudness"]
    
    async def close(self):
        """Mock close implementation."""
        pass
    
    # Helper methods for testing
    
    def add_sound(self, sound: SoundMetadata):
        """Add a sound to the mock repository."""
        self._sounds[sound.id] = sound
    
    def set_search_results(self, query: str, sounds: list):
        """Set search results for a query."""
        self._search_results[query] = sounds


class TestSoundMetadata:
    """Test SoundMetadata model."""
    
    def test_creation_with_required_fields(self):
        """Test creating sound metadata with required fields."""
        sound = SoundMetadata(
            id="123",
            name="Test Sound",
            provider="test"
        )
        
        assert sound.id == "123"
        assert sound.name == "Test Sound"
        assert sound.provider == "test"
        assert sound.tags == []
        assert sound.extra_metadata == {}
    
    def test_creation_with_all_fields(self):
        """Test creating sound metadata with all fields."""
        created_at = datetime.now()
        
        sound = SoundMetadata(
            id="456",
            name="Complete Sound",
            provider="freesound",
            description="A test sound",
            tags=["test", "music"],
            license="CC0",
            created_at=created_at,
            duration=10.5,
            sample_rate=44100,
            channels=2,
            file_format="wav",
            file_size=1024000,
            url="https://example.com/sound",
            download_url="https://example.com/download",
            preview_url="https://example.com/preview",
            mir_features={"spectral_centroid": 1500.0},
            extra_metadata={"user": "testuser"}
        )
        
        assert sound.duration == 10.5
        assert sound.sample_rate == 44100
        assert sound.tags == ["test", "music"]
        assert sound.mir_features["spectral_centroid"] == 1500.0
    
    def test_validation_empty_id(self):
        """Test validation fails with empty ID."""
        with pytest.raises(ValueError, match="Sound ID is required"):
            SoundMetadata(id="", name="Test", provider="test")
    
    def test_validation_empty_provider(self):
        """Test validation fails with empty provider."""
        with pytest.raises(ValueError, match="Provider is required"):
            SoundMetadata(id="123", name="Test", provider="")
    
    def test_duration_formatted(self):
        """Test formatted duration property."""
        sound = SoundMetadata(id="1", name="Test", provider="test", duration=125.5)
        assert sound.duration_formatted == "02:05"
        
        sound_no_duration = SoundMetadata(id="2", name="Test", provider="test")
        assert sound_no_duration.duration_formatted == "Unknown"
    
    def test_quality_score(self):
        """Test quality score calculation."""
        # High quality sound
        high_quality = SoundMetadata(
            id="1",
            name="High Quality",
            provider="freesound",
            duration=15.0,  # Good duration
            sample_rate=44100,  # Good sample rate
            file_format="wav",  # Lossless format
            description="A detailed description",
            tags=["music", "test"],
            license="CC0",
            extra_metadata={"num_ratings": 10, "avg_rating": 4.5}
        )
        
        # Low quality sound
        low_quality = SoundMetadata(
            id="2",
            name="Low Quality",
            provider="test"
        )
        
        assert high_quality.quality_score > low_quality.quality_score
        assert high_quality.quality_score > 50  # Should be reasonably high
    
    def test_mir_feature_access(self):
        """Test MIR feature access methods."""
        sound = SoundMetadata(
            id="1",
            name="Test",
            provider="test",
            mir_features={
                "lowlevel": {
                    "spectral_centroid": 1500.0
                },
                "rhythm": {
                    "bpm": 120.0
                }
            }
        )
        
        # Test has_mir_feature
        assert sound.has_mir_feature("lowlevel.spectral_centroid")
        assert sound.has_mir_feature("rhythm.bpm")
        assert not sound.has_mir_feature("nonexistent.feature")
        
        # Test get_mir_feature
        assert sound.get_mir_feature("lowlevel.spectral_centroid") == 1500.0
        assert sound.get_mir_feature("rhythm.bpm") == 120.0
        assert sound.get_mir_feature("nonexistent.feature", "default") == "default"
    
    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        original = SoundMetadata(
            id="123",
            name="Test Sound",
            provider="test",
            duration=10.0,
            tags=["test"]
        )
        
        # Test serialization
        data = original.to_dict()
        assert data["id"] == "123"
        assert data["name"] == "Test Sound"
        assert data["duration"] == 10.0
        
        # Test deserialization
        restored = SoundMetadata.from_dict(data)
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.duration == original.duration
        assert restored.tags == original.tags


class TestSearchModels:
    """Test search request and response models."""
    
    def test_search_request_creation(self):
        """Test creating search request."""
        request = SearchRequest(
            query="piano",
            limit=20,
            offset=10,
            filters={"duration_min": 5.0},
            sort_by="relevance"
        )
        
        assert request.query == "piano"
        assert request.limit == 20
        assert request.offset == 10
        assert request.filters["duration_min"] == 5.0
    
    def test_search_request_validation(self):
        """Test search request validation."""
        # Empty query
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            SearchRequest(query="")
        
        # Negative limit
        with pytest.raises(ValueError, match="Limit must be positive"):
            SearchRequest(query="test", limit=0)
        
        # Negative offset
        with pytest.raises(ValueError, match="Offset cannot be negative"):
            SearchRequest(query="test", offset=-1)
    
    def test_search_response_creation(self):
        """Test creating search response."""
        sounds = [
            SoundMetadata(id="1", name="Sound 1", provider="test"),
            SoundMetadata(id="2", name="Sound 2", provider="test")
        ]
        
        response = SearchResponse(
            sounds=sounds,
            total_count=10,
            query="test",
            provider="test",
            limit=2,
            offset=0
        )
        
        assert len(response.sounds) == 2
        assert response.total_count == 10
        assert response.has_more_results is True
        assert response.next_offset == 2
    
    def test_search_response_pagination(self):
        """Test search response pagination properties."""
        response = SearchResponse(
            sounds=[SoundMetadata(id="1", name="Test", provider="test")],
            total_count=25,
            query="test",
            provider="test",
            limit=10,
            offset=20
        )
        
        page_info = response.page_info
        assert page_info["current_page"] == 3
        assert page_info["total_pages"] == 3
        assert page_info["has_previous"] is True
        assert page_info["has_next"] is False
    
    def test_search_response_filtering(self):
        """Test search response filtering methods."""
        sounds = [
            SoundMetadata(id="1", name="Short", provider="test", duration=3.0, tags=["short"]),
            SoundMetadata(id="2", name="Long", provider="test", duration=30.0, tags=["long"]),
            SoundMetadata(id="3", name="Medium", provider="test", duration=15.0, tags=["medium"])
        ]
        
        response = SearchResponse(
            sounds=sounds,
            total_count=3,
            query="test",
            provider="test"
        )
        
        # Test duration filtering
        filtered = response.filter_by_duration(10.0, 20.0)
        assert len(filtered.sounds) == 1
        assert filtered.sounds[0].name == "Medium"
        
        # Test tag filtering
        tag_filtered = response.filter_by_tags(["short"])
        assert len(tag_filtered.sounds) == 1
        assert tag_filtered.sounds[0].name == "Short"


class TestDatabaseService:
    """Test DatabaseService functionality."""
    
    @pytest.fixture
    def mock_providers(self):
        """Create mock providers for testing."""
        provider1 = MockRepository("provider1")
        provider2 = MockRepository("provider2")
        
        # Add test sounds
        sound1 = SoundMetadata(id="1", name="Test Sound 1", provider="provider1")
        sound2 = SoundMetadata(id="2", name="Test Sound 2", provider="provider2")
        
        provider1.add_sound(sound1)
        provider2.add_sound(sound2)
        
        # Set search results
        provider1.set_search_results("piano", [sound1])
        provider2.set_search_results("piano", [sound2])
        
        return {"provider1": provider1, "provider2": provider2}
    
    @pytest.fixture
    def database_service(self, mock_providers):
        """Create database service with mock providers."""
        return DatabaseService(
            providers=mock_providers,
            default_provider="provider1"
        )
    
    def test_initialization(self, mock_providers):
        """Test database service initialization."""
        service = DatabaseService(
            providers=mock_providers,
            default_provider="provider1"
        )
        
        assert service.default_provider_name == "provider1"
        assert len(service.providers) == 2
        assert service.default_provider.name == "provider1"
    
    def test_invalid_default_provider(self, mock_providers):
        """Test initialization with invalid default provider."""
        with pytest.raises(ValueError, match="Default provider 'invalid' not found"):
            DatabaseService(
                providers=mock_providers,
                default_provider="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_search_default_provider(self, database_service):
        """Test search using default provider."""
        response = await database_service.search("piano", limit=5)
        
        assert response.query == "piano"
        assert response.provider == "provider1"
        assert len(response.sounds) == 1
        assert response.sounds[0].name == "Test Sound 1"
    
    @pytest.mark.asyncio
    async def test_search_specific_provider(self, database_service):
        """Test search using specific provider."""
        response = await database_service.search("piano", provider="provider2")
        
        assert response.provider == "provider2"
        assert len(response.sounds) == 1
        assert response.sounds[0].name == "Test Sound 2"
    
    @pytest.mark.asyncio
    async def test_search_invalid_provider(self, database_service):
        """Test search with invalid provider."""
        with pytest.raises(ProviderError, match="Provider not found"):
            await database_service.search("piano", provider="invalid")
    
    @pytest.mark.asyncio
    async def test_search_merge_providers(self, database_service):
        """Test search merging results from all providers."""
        response = await database_service.search("piano", merge_providers=True)
        
        assert len(response.sounds) == 2  # Results from both providers
        assert "provider1,provider2" in response.provider
    
    @pytest.mark.asyncio
    async def test_get_by_id_default_provider(self, database_service):
        """Test get by ID using default provider."""
        sound = await database_service.get_by_id("1")
        
        assert sound is not None
        assert sound.id == "1"
        assert sound.name == "Test Sound 1"
    
    @pytest.mark.asyncio
    async def test_get_by_id_specific_provider(self, database_service):
        """Test get by ID using specific provider."""
        sound = await database_service.get_by_id("2", provider="provider2")
        
        assert sound is not None
        assert sound.id == "2"
        assert sound.provider == "provider2"
    
    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, database_service):
        """Test get by ID when sound not found."""
        sound = await database_service.get_by_id("nonexistent")
        assert sound is None
    
    @pytest.mark.asyncio
    async def test_download(self, database_service, tmp_path):
        """Test sound download."""
        output_path = tmp_path / "test_sound.wav"
        
        result_path = await database_service.download("1", output_path)
        
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.read_text() == "mock audio data"
    
    @pytest.mark.asyncio
    async def test_download_not_found(self, database_service, tmp_path):
        """Test download when sound not found."""
        output_path = tmp_path / "test_sound.wav"
        
        with pytest.raises(SoundNotFoundError):
            await database_service.download("nonexistent", output_path)
    
    def test_list_providers(self, database_service):
        """Test listing providers."""
        providers = database_service.list_providers()
        
        assert len(providers) == 2
        provider_names = [p.name for p in providers]
        assert "provider1" in provider_names
        assert "provider2" in provider_names
        
        # Check default provider marking
        default_provider = next(p for p in providers if p.is_default)
        assert default_provider.name == "provider1"
    
    def test_get_available_providers(self, database_service):
        """Test getting available providers."""
        available = database_service.get_available_providers()
        
        assert len(available) == 2
        assert "provider1" in available
        assert "provider2" in available
    
    @pytest.mark.asyncio
    async def test_health_check(self, database_service):
        """Test health check functionality."""
        health = await database_service.health_check()
        
        assert health["overall_status"] == "healthy"
        assert health["healthy_providers"] == 2
        assert health["total_providers"] == 2
        assert "provider1" in health["providers"]
        assert "provider2" in health["providers"]
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_providers):
        """Test database service as async context manager."""
        async with DatabaseService(mock_providers) as service:
            assert service is not None
            # Service should be usable within context
            providers = service.list_providers()
            assert len(providers) == 2