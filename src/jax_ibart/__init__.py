"""jax-ibart: a JAX-native reimplementation of iBART.

iBART (iterative Bayesian Additive Regression Trees) discovers compact,
interpretable nonlinear descriptors by interleaving feature engineering with
nonparametric feature selection. References: Liu et al. 2022 (J. Chem. Phys.
156:164105); Ye, Senftle & Li 2024 (JASA 119:81-94).
"""

from .ibart import IBart
from .selection import BartGSESelector, DiffSelector, make_selector
from . import uncertainty, dynamics

__all__ = ["IBart", "BartGSESelector", "DiffSelector", "make_selector",
           "uncertainty", "dynamics"]
__version__ = "0.0.1"
