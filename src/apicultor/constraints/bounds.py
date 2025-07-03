#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Union
import numpy as np
from numpy.typing import NDArray


def dsvm_low_a(a: NDArray[np.floating]) -> NDArray[np.floating]:
    """Apply lower bound constraint to array values.
    
    Args:
        a: Input array to constrain
        
    Returns:
        Array with negative values set to 0.0
    """
    a[a < 0] = 0.0
    return a


def dsvm_high_a(a: NDArray[np.floating], cw: NDArray[np.floating], c: float) -> NDArray[np.floating]:
    """Apply upper bound constraint to array values.
    
    Args:
        a: Input array to constrain
        cw: Weight array for constraint calculation
        c: Constraint multiplier
        
    Returns:
        Array with upper bounds applied based on constraint formula
    """
    a = np.array([min(a[i], c * cw[i] * cw[i]) if a[i] > 0 else a[i] for i in range(len(a))])
    return a


def es(a: NDArray[np.floating], lab: NDArray[np.floating], features: NDArray[np.floating]) -> Union[float, int]:
    """Calculate error statistic for optimization.
    
    Args:
        a: Alpha values array
        lab: Label values array
        features: Feature matrix
        
    Returns:
        Median error statistic or 0 if no positive alpha values
    """
    if any(a > 0.0):                                               
        return np.median((lab[a > 0.0] - np.sum(a[a > 0.0] * lab[a > 0.0] * features[a > 0.0].T).T * features[a > 0.0].T).T)
    else:
        return 0
