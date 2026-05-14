"""
Compatibility wrapper for Phase 5 forgetting dynamics.
"""

from .forgetting_dynamics import ForgettingDynamics


ForgettingModule = ForgettingDynamics

__all__ = ["ForgettingDynamics", "ForgettingModule"]
