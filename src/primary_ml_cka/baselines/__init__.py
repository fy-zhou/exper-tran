"""Independent, auditable baselines for KDD 2027 experiments.

Only algorithms explicitly implemented and tested are exported.  Published
multimodal methods must not be represented by a generic PGD substitute.
"""
from .iterative import AttackConfig, GradientAttack, attack_image

__all__ = ["AttackConfig", "GradientAttack", "attack_image"]
