"""Test cases for constraint modules."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from apicultor.constraints.bounds import dsvm_low_a, dsvm_high_a, es
from apicultor.constraints.tempo import same_time


class TestBounds:
    """Test bounds constraint functions."""
    
    def test_dsvm_low_a_positive_values(self):
        """Test dsvm_low_a with positive values."""
        a = np.array([1.0, 2.0, 3.0])
        result = dsvm_low_a(a)
        assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    
    def test_dsvm_low_a_negative_values(self):
        """Test dsvm_low_a with negative values."""
        a = np.array([-1.0, -2.0, -3.0])
        result = dsvm_low_a(a)
        assert_array_equal(result, np.array([0.0, 0.0, 0.0]))
    
    def test_dsvm_low_a_mixed_values(self):
        """Test dsvm_low_a with mixed positive and negative values."""
        a = np.array([-1.0, 0.0, 1.0, -2.0, 3.0])
        result = dsvm_low_a(a)
        assert_array_equal(result, np.array([0.0, 0.0, 1.0, 0.0, 3.0]))
    
    def test_dsvm_low_a_empty_array(self):
        """Test dsvm_low_a with empty array."""
        a = np.array([])
        result = dsvm_low_a(a)
        assert_array_equal(result, np.array([]))
    
    def test_dsvm_high_a_basic_case(self):
        """Test dsvm_high_a with basic case."""
        a = np.array([1.0, 2.0, 3.0])
        cw = np.array([0.5, 0.5, 0.5])
        c = 2.0
        result = dsvm_high_a(a, cw, c)
        expected = np.array([0.5, 0.5, 0.5])  # min(a[i], c * cw[i]^2)
        assert_array_equal(result, expected)
    
    def test_dsvm_high_a_negative_values(self):
        """Test dsvm_high_a preserves negative values."""
        a = np.array([-1.0, 2.0, -3.0])
        cw = np.array([0.5, 0.5, 0.5])
        c = 2.0
        result = dsvm_high_a(a, cw, c)
        expected = np.array([-1.0, 0.5, -3.0])  # negative values preserved
        assert_array_equal(result, expected)
    
    def test_dsvm_high_a_large_constraint(self):
        """Test dsvm_high_a with large constraint values."""
        a = np.array([1.0, 2.0, 3.0])
        cw = np.array([2.0, 2.0, 2.0])
        c = 10.0
        result = dsvm_high_a(a, cw, c)
        expected = np.array([1.0, 2.0, 3.0])  # original values preserved
        assert_array_equal(result, expected)
    
    def test_es_with_positive_alpha(self):
        """Test es function with positive alpha values."""
        a = np.array([1.0, 0.0, 1.0])
        lab = np.array([1.0, 2.0, 3.0])
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        result = es(a, lab, features)
        assert isinstance(result, (int, float, np.number))
    
    def test_es_with_all_zero_alpha(self):
        """Test es function with all zero alpha values."""
        a = np.array([0.0, 0.0, 0.0])
        lab = np.array([1.0, 2.0, 3.0])
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        result = es(a, lab, features)
        assert result == 0
    
    def test_es_with_negative_alpha(self):
        """Test es function with negative alpha values."""
        a = np.array([-1.0, -2.0, -3.0])
        lab = np.array([1.0, 2.0, 3.0])
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        result = es(a, lab, features)
        assert result == 0
    
    def test_es_with_mixed_alpha(self):
        """Test es function with mixed positive/negative alpha values."""
        a = np.array([1.0, -1.0, 2.0])
        lab = np.array([1.0, 2.0, 3.0])
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        result = es(a, lab, features)
        assert isinstance(result, (int, float, np.number))


class TestTempo:
    """Test tempo constraint functions."""
    
    def test_same_time_equal_tempo(self):
        """Test same_time with signals of equal tempo."""
        # Create mock signals (we'll need to mock librosa for actual testing)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        
        # Mock librosa.beat.beat_track to return same tempo
        import librosa
        original_beat_track = librosa.beat.beat_track
        
        def mock_beat_track(audio, **kwargs):
            return 120.0, np.array([])  # Same tempo for both
        
        librosa.beat.beat_track = mock_beat_track
        
        try:
            result_x, result_y = same_time(x, y)
            assert len(result_x) == len(result_y)
            assert result_x.dtype == result_y.dtype
        finally:
            librosa.beat.beat_track = original_beat_track
    
    def test_same_time_different_tempo(self):
        """Test same_time with signals of different tempo."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        
        # Mock librosa functions
        import librosa
        original_beat_track = librosa.beat.beat_track
        original_time_stretch = librosa.effects.time_stretch
        
        call_count = 0
        def mock_beat_track(audio, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 120.0, np.array([])  # First call (x)
            else:
                return 140.0, np.array([])  # Second call (y)
        
        def mock_time_stretch(audio, rate, **kwargs):
            # Simple mock that returns modified length array
            new_length = int(len(audio) * rate)
            return np.resize(audio, new_length)
        
        librosa.beat.beat_track = mock_beat_track
        librosa.effects.time_stretch = mock_time_stretch
        
        try:
            result_x, result_y = same_time(x, y)
            assert len(result_x) == len(result_y)
            assert result_x.dtype == result_y.dtype
        finally:
            librosa.beat.beat_track = original_beat_track
            librosa.effects.time_stretch = original_time_stretch
    
    def test_same_time_empty_arrays(self):
        """Test same_time with empty arrays."""
        x = np.array([])
        y = np.array([])
        
        # Mock librosa to handle empty arrays
        import librosa
        original_beat_track = librosa.beat.beat_track
        
        def mock_beat_track(audio, **kwargs):
            if len(audio) == 0:
                return 120.0, np.array([])
            return 120.0, np.array([])
        
        librosa.beat.beat_track = mock_beat_track
        
        try:
            result_x, result_y = same_time(x, y)
            assert len(result_x) == len(result_y)
            assert len(result_x) == 0
        finally:
            librosa.beat.beat_track = original_beat_track
    
    def test_same_time_single_sample(self):
        """Test same_time with single sample arrays."""
        x = np.array([1.0])
        y = np.array([2.0])
        
        # Mock librosa for single sample
        import librosa
        original_beat_track = librosa.beat.beat_track
        original_time_stretch = librosa.effects.time_stretch
        
        def mock_beat_track(audio, **kwargs):
            return 120.0, np.array([])
        
        def mock_time_stretch(audio, rate, **kwargs):
            return audio  # No change for single sample
        
        librosa.beat.beat_track = mock_beat_track
        librosa.effects.time_stretch = mock_time_stretch
        
        try:
            result_x, result_y = same_time(x, y)
            assert len(result_x) == len(result_y)
            assert len(result_x) == 1
        finally:
            librosa.beat.beat_track = original_beat_track
            librosa.effects.time_stretch = original_time_stretch


class TestConstraintsIntegration:
    """Integration tests for constraint functions."""
    
    def test_bounds_pipeline(self):
        """Test applying bounds constraints in sequence."""
        a = np.array([-1.0, 2.0, -3.0, 4.0])
        cw = np.array([0.5, 0.5, 0.5, 0.5])
        c = 2.0
        
        # Apply lower bound first
        a_low = dsvm_low_a(a.copy())
        expected_low = np.array([0.0, 2.0, 0.0, 4.0])
        assert_array_equal(a_low, expected_low)
        
        # Apply upper bound
        a_high = dsvm_high_a(a_low, cw, c)
        expected_high = np.array([0.0, 0.5, 0.0, 0.5])
        assert_array_equal(a_high, expected_high)
    
    def test_bounds_with_optimization_context(self):
        """Test bounds functions in optimization context."""
        # Simulate optimization variables
        n_samples = 10
        a = np.random.randn(n_samples)
        cw = np.random.rand(n_samples)
        c = 1.0
        
        # Test that bounds are properly applied
        a_constrained = dsvm_low_a(a.copy())
        assert np.all(a_constrained >= 0)
        
        a_constrained = dsvm_high_a(a_constrained, cw, c)
        positive_mask = a_constrained > 0
        expected_upper = c * cw[positive_mask] ** 2
        assert np.all(a_constrained[positive_mask] <= expected_upper)
    
    def test_error_handling(self):
        """Test error handling in constraint functions."""
        # Test with incompatible array sizes
        a = np.array([1.0, 2.0])
        cw = np.array([0.5])  # Different size
        c = 1.0
        
        with pytest.raises(IndexError):
            dsvm_high_a(a, cw, c)
        
        # Test with invalid input types
        with pytest.raises((TypeError, AttributeError)):
            dsvm_low_a("not_an_array")
    
    def test_numerical_stability(self):
        """Test numerical stability of constraint functions."""
        # Test with very small values
        a = np.array([1e-10, -1e-10, 1e-15])
        result = dsvm_low_a(a)
        assert result[0] > 0
        assert result[1] == 0
        assert result[2] >= 0
        
        # Test with very large values
        a = np.array([1e10, -1e10, 1e15])
        cw = np.array([1e-5, 1e-5, 1e-5])
        c = 1e5
        result = dsvm_high_a(a, cw, c)
        assert np.all(np.isfinite(result))