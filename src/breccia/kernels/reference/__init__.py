"""Reference (slow but correct) kernels for cast, dequantize, and matmul.

Every optimized kernel must produce numerically equivalent output to
these references within the recipe's declared tolerance.
"""

from .cast import cast, dequantize, requantize
from .matmul import matmul

__all__ = ["cast", "dequantize", "requantize", "matmul"]
