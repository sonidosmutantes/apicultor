"""Core interfaces for apicultor plugin system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


class PluginInterface(ABC):
    """Base interface for all apicultor plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name identifier."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """List of required plugin dependencies."""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration.
        
        Args:
            config: Plugin-specific configuration dictionary
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Clean up plugin resources."""
        pass
    
    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if plugin is currently enabled."""
        pass


class AudioProcessorInterface(PluginInterface):
    """Interface for audio processing plugins."""
    
    @abstractmethod
    def process_audio(
        self, 
        audio: NDArray[np.floating], 
        sample_rate: int,
        **kwargs: Any
    ) -> NDArray[np.floating]:
        """Process audio data.
        
        Args:
            audio: Input audio signal
            sample_rate: Audio sample rate
            **kwargs: Additional processing parameters
            
        Returns:
            Processed audio signal
        """
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of supported audio formats."""
        pass
    
    @property
    @abstractmethod
    def required_sample_rates(self) -> List[int]:
        """List of required sample rates (empty means any)."""
        pass


class DatabaseInterface(PluginInterface):
    """Interface for database access plugins."""
    
    @abstractmethod
    def search_sounds(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for sounds in the database.
        
        Args:
            query: Search query string
            filters: Optional search filters
            limit: Maximum number of results
            
        Returns:
            List of sound metadata dictionaries
        """
        pass
    
    @abstractmethod
    def get_sound_by_id(self, sound_id: str) -> Optional[Dict[str, Any]]:
        """Get sound metadata by ID.
        
        Args:
            sound_id: Unique sound identifier
            
        Returns:
            Sound metadata dictionary or None if not found
        """
        pass
    
    @abstractmethod
    def download_sound(self, sound_id: str, output_path: Path) -> bool:
        """Download sound file.
        
        Args:
            sound_id: Unique sound identifier
            output_path: Path to save the downloaded file
            
        Returns:
            True if download successful, False otherwise
        """
        pass


class MIRAnalysisInterface(PluginInterface):
    """Interface for MIR analysis plugins."""
    
    @abstractmethod
    def extract_features(
        self, 
        audio: NDArray[np.floating], 
        sample_rate: int,
        descriptors: List[str]
    ) -> Dict[str, Any]:
        """Extract MIR features from audio.
        
        Args:
            audio: Input audio signal
            sample_rate: Audio sample rate
            descriptors: List of descriptor names to extract
            
        Returns:
            Dictionary mapping descriptor names to values
        """
        pass
    
    @property
    @abstractmethod
    def available_descriptors(self) -> List[str]:
        """List of available MIR descriptors."""
        pass


class MachineLearningInterface(PluginInterface):
    """Interface for machine learning plugins."""
    
    @abstractmethod
    def train_model(
        self, 
        features: NDArray[np.floating], 
        labels: Optional[NDArray[np.floating]] = None,
        **kwargs: Any
    ) -> Any:
        """Train a machine learning model.
        
        Args:
            features: Feature matrix
            labels: Optional target labels for supervised learning
            **kwargs: Additional training parameters
            
        Returns:
            Trained model object
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        model: Any, 
        features: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Make predictions using trained model.
        
        Args:
            model: Trained model object
            features: Feature matrix for prediction
            
        Returns:
            Prediction results
        """
        pass


class SegmentationInterface(PluginInterface):
    """Interface for audio segmentation plugins."""
    
    @abstractmethod
    def segment_audio(
        self, 
        audio: NDArray[np.floating], 
        sample_rate: int,
        **kwargs: Any
    ) -> List[tuple]:
        """Segment audio into regions.
        
        Args:
            audio: Input audio signal
            sample_rate: Audio sample rate
            **kwargs: Segmentation parameters
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        pass
    
    @property
    @abstractmethod
    def segmentation_method(self) -> str:
        """Name of the segmentation method."""
        pass


class EmotionAnalysisInterface(PluginInterface):
    """Interface for music emotion analysis plugins."""
    
    @abstractmethod
    def analyze_emotion(
        self, 
        audio: NDArray[np.floating], 
        sample_rate: int
    ) -> Dict[str, float]:
        """Analyze emotional content of audio.
        
        Args:
            audio: Input audio signal
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary mapping emotion names to confidence scores
        """
        pass
    
    @property
    @abstractmethod
    def emotion_categories(self) -> List[str]:
        """List of supported emotion categories."""
        pass


class StateMachineInterface(PluginInterface):
    """Interface for state machine composition plugins."""
    
    @abstractmethod
    def create_composition(
        self, 
        sounds: List[Dict[str, Any]], 
        transitions: Dict[str, Any]
    ) -> Any:
        """Create a state machine composition.
        
        Args:
            sounds: List of sound metadata
            transitions: Transition probability matrix or rules
            
        Returns:
            Composition object
        """
        pass
    
    @abstractmethod
    def generate_sequence(
        self, 
        composition: Any, 
        length: int
    ) -> List[str]:
        """Generate a sequence from the composition.
        
        Args:
            composition: Composition object
            length: Desired sequence length
            
        Returns:
            List of sound IDs in the generated sequence
        """
        pass


class SonificationInterface(PluginInterface):
    """Interface for sonification plugins."""
    
    @abstractmethod
    def sonify_data(
        self, 
        data: NDArray[np.floating], 
        mapping: Dict[str, Any],
        duration: float = 5.0
    ) -> NDArray[np.floating]:
        """Convert data to audio through sonification.
        
        Args:
            data: Input data to sonify
            mapping: Parameter mapping configuration
            duration: Output audio duration in seconds
            
        Returns:
            Generated audio signal
        """
        pass
    
    @property
    @abstractmethod
    def supported_mappings(self) -> List[str]:
        """List of supported parameter mappings."""
        pass