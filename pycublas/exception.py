import os
import sys


# Generic CUBLAS error:
class cublasError(Exception):
    """CUBLAS error"""

    pass


# Exceptions corresponding to different CUBLAS errors:
class cublasNotInitialized(cublasError):
    """CUBLAS library not initialized."""

    pass


class cublasAllocFailed(cublasError):
    """Resource allocation failed."""

    pass


class cublasInvalidValue(cublasError):
    """Unsupported numerical value was passed to function."""

    pass


class cublasArchMismatch(cublasError):
    """Function requires an architectural feature absent from the device."""

    pass


class cublasMappingError(cublasError):
    """Access to GPU memory space failed."""

    pass


class cublasExecutionFailed(cublasError):
    """GPU program failed to execute."""

    pass


class cublasInternalError(cublasError):
    """An internal CUBLAS operation failed."""

    pass


class cublasNotSupported(cublasError):
    """Not supported."""

    pass


class cublasLicenseError(cublasError):
    """License error."""

    pass


cublasExceptions = {
    1: cublasNotInitialized,
    3: cublasAllocFailed,
    7: cublasInvalidValue,
    8: cublasArchMismatch,
    11: cublasMappingError,
    13: cublasExecutionFailed,
    14: cublasInternalError,
    15: cublasNotSupported,
    16: cublasLicenseError,
}
