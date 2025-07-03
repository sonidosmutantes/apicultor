"""Constraints plugin implementation."""

from typing import Any, Dict, List, Union
import numpy as np
from numpy.typing import NDArray
import logging

from ..core.interfaces import PluginInterface
from ..constraints.bounds import dsvm_low_a, dsvm_high_a, es
from ..constraints.tempo import same_time


logger = logging.getLogger(__name__)


class ConstraintsPlugin(PluginInterface):
    """Plugin for mathematical constraints and optimization utilities."""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._enabled = False
        self._tolerance = 1e-10
        self._max_iterations = 1000
    
    @property
    def name(self) -> str:
        """Plugin name identifier."""
        return "constraints"
    
    @property
    def version(self) -> str:
        """Plugin version."""
        return "1.0.0"
    
    @property
    def description(self) -> str:
        """Plugin description."""
        return "Mathematical constraints and optimization utilities for audio processing"
    
    @property
    def dependencies(self) -> List[str]:
        """List of required plugin dependencies."""
        return []  # Constraints is a mathematical utility with no dependencies
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration.
        
        Args:
            config: Plugin-specific configuration dictionary
        """
        self._config = config
        self._tolerance = config.get("tolerance", 1e-10)
        self._max_iterations = config.get("max_iterations", 1000)
        
        # Validate configuration
        if self._tolerance <= 0:
            raise ValueError("Tolerance must be positive")
        if self._max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        
        self._enabled = True
        logger.info(f"Constraints plugin initialized (tolerance={self._tolerance}, max_iter={self._max_iterations})")
    
    def shutdown(self) -> None:
        """Clean up plugin resources."""
        self._enabled = False
        logger.info("Constraints plugin shutdown complete")
    
    @property
    def is_enabled(self) -> bool:
        """Check if plugin is currently enabled."""
        return self._enabled
    
    def apply_lower_bounds(self, values: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply lower bound constraints (zero threshold).
        
        Args:
            values: Input array to constrain
            
        Returns:
            Array with negative values set to 0.0
        """
        if not self._enabled:
            raise RuntimeError("Constraints plugin not initialized")
        
        return dsvm_low_a(values)
    
    def apply_upper_bounds(
        self, 
        values: NDArray[np.floating], 
        weights: NDArray[np.floating], 
        multiplier: float
    ) -> NDArray[np.floating]:
        """Apply upper bound constraints.
        
        Args:
            values: Input array to constrain
            weights: Weight array for constraint calculation
            multiplier: Constraint multiplier
            
        Returns:
            Array with upper bounds applied
        """
        if not self._enabled:
            raise RuntimeError("Constraints plugin not initialized")
        
        if len(values) != len(weights):
            raise ValueError("Values and weights arrays must have same length")
        
        return dsvm_high_a(values, weights, multiplier)
    
    def calculate_error_statistic(
        self, 
        alpha: NDArray[np.floating], 
        labels: NDArray[np.floating], 
        features: NDArray[np.floating]
    ) -> Union[float, int]:
        """Calculate error statistic for optimization.
        
        Args:
            alpha: Alpha values array
            labels: Label values array  
            features: Feature matrix
            
        Returns:
            Error statistic value
        """
        if not self._enabled:
            raise RuntimeError("Constraints plugin not initialized")
        
        if len(alpha) != len(labels):
            raise ValueError("Alpha and labels arrays must have same length")
        if len(alpha) != features.shape[0]:
            raise ValueError("Alpha array length must match feature matrix rows")
        
        return es(alpha, labels, features)
    
    def synchronize_tempo(
        self, 
        audio1: NDArray[np.floating], 
        audio2: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Synchronize two audio signals to have the same tempo.
        
        Args:
            audio1: First audio signal
            audio2: Second audio signal
            
        Returns:
            Tuple of synchronized audio signals
        """
        if not self._enabled:
            raise RuntimeError("Constraints plugin not initialized")
        
        if audio1.size == 0 or audio2.size == 0:
            logger.warning("Empty audio array provided to tempo synchronization")
            return audio1, audio2
        
        try:
            return same_time(audio1, audio2)
        except Exception as e:
            logger.error(f"Tempo synchronization failed: {e}")
            # Return original arrays on failure
            return audio1, audio2
    
    def apply_constraint_pipeline(
        self,
        values: NDArray[np.floating],
        weights: Optional[NDArray[np.floating]] = None,
        multiplier: float = 1.0,
        apply_lower: bool = True,
        apply_upper: bool = True
    ) -> NDArray[np.floating]:
        """Apply a complete constraint pipeline.
        
        Args:
            values: Input array to constrain
            weights: Optional weight array for upper bounds
            multiplier: Constraint multiplier for upper bounds
            apply_lower: Whether to apply lower bound constraints
            apply_upper: Whether to apply upper bound constraints
            
        Returns:
            Fully constrained array
        """
        if not self._enabled:
            raise RuntimeError("Constraints plugin not initialized")
        
        result = values.copy()
        
        # Apply lower bounds
        if apply_lower:
            result = self.apply_lower_bounds(result)
        
        # Apply upper bounds
        if apply_upper and weights is not None:
            result = self.apply_upper_bounds(result, weights, multiplier)
        
        return result
    
    def validate_constraints(
        self,
        values: NDArray[np.floating],
        lower_bound: float = 0.0,
        upper_bound: Optional[float] = None
    ) -> Dict[str, bool]:
        """Validate that values satisfy constraints.
        
        Args:
            values: Array to validate
            lower_bound: Minimum allowed value
            upper_bound: Maximum allowed value (optional)
            
        Returns:
            Dictionary with validation results
        """
        if not self._enabled:
            raise RuntimeError("Constraints plugin not initialized")
        
        results = {
            "lower_bound_satisfied": np.all(values >= lower_bound - self._tolerance),
            "finite_values": np.all(np.isfinite(values)),
            "no_nan_values": not np.any(np.isnan(values)),
        }
        
        if upper_bound is not None:
            results["upper_bound_satisfied"] = np.all(values <= upper_bound + self._tolerance)
        
        results["all_constraints_satisfied"] = all(results.values())
        
        return results
    
    def get_constraint_info(self) -> Dict[str, Any]:
        """Get information about current constraint settings.
        
        Returns:
            Dictionary with constraint configuration
        """
        return {
            "tolerance": self._tolerance,
            "max_iterations": self._max_iterations,
            "enabled": self._enabled,
            "available_methods": [
                "apply_lower_bounds",
                "apply_upper_bounds", 
                "calculate_error_statistic",
                "synchronize_tempo",
                "apply_constraint_pipeline",
                "validate_constraints"
            ]
        }