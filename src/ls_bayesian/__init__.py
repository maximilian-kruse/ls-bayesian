"""A modular toolbox for large-scale Bayesian inverse problems.

All type hints in the package are enforced at runtime via beartype, which also validates the
`beartype.vale` constraints in `Annotated` hints. Arguments violating a hint raise a
`beartype.roar.BeartypeCallHintViolation`.
"""

from beartype.claw import beartype_this_package

from ls_bayesian._version import __version__

beartype_this_package()

__all__ = ["__version__"]
