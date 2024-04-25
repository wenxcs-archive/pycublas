# This is a tailored version modified from scikit-cuda project
# under the BSD License.
import ctypes
import os
import sys
import pycublas.exception as exceptions
import pycublas.constant as constant


class CublasInterop:
    class _types:
        handle = ctypes.c_void_p
        stream = ctypes.c_void_p

    def cublasCheckStatus(self, status):
        if status != 0:
            try:
                e = exceptions.cublasExceptions[status]
                raise e
            except KeyError:
                raise exceptions.cublasError

    def __init__(self, cublas_libname: str = "libcublas.so"):
        if "linux" not in sys.platform:
            raise RuntimeError("unsupported platform")

        self._libcublas_libname = cublas_libname

        if "CUBLAS_LIBNAME" in os.environ:
            self._libcublas_libname = os.environ["CUBLAS_LIBNAME"]

        self._libcublas = ctypes.cdll.LoadLibrary(self._libcublas_libname)
        if self._libcublas == None:
            raise OSError("cublas library not found")

        # cublasCreate_v2
        self._libcublas.cublasCreate_v2.restype = int
        self._libcublas.cublasCreate_v2.argtypes = [self._types.handle]

        # cublasDestroy_v2
        self._libcublas.cublasDestroy_v2.restype = int
        self._libcublas.cublasDestroy_v2.argtypes = [self._types.handle]

        # cublasSgemm_v2
        self._libcublas.cublasSgemm_v2.restype = int
        self._libcublas.cublasSgemm_v2.argtypes = [
            self._types.handle,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        # cublasHgemm
        self._libcublas.cublasHgemm.restype = int
        self._libcublas.cublasHgemm.argtypes = [
            self._types.handle,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        # cublasGemmEx
        self._libcublas.cublasGemmEx.restype = int
        self._libcublas.cublasGemmEx.argtypes = [
            self._types.handle,
            ctypes.c_int,  # trans
            ctypes.c_int,
            ctypes.c_int,  # m,n,k
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,  # alpha
            ctypes.c_void_p,  # A
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,  # B
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,  # beta
            ctypes.c_void_p,  # C
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,  # computeType
            ctypes.c_int,  # algo
        ]

    def cublasCreate_v2(self):
        handle = self._types.handle()
        status = self._libcublas.cublasCreate_v2(ctypes.byref(handle))
        self.cublasCheckStatus(status)
        return handle.value

    def cublasDestroy_v2(self, handle):
        status = self._libcublas.cublasDestroy_v2(handle)
        self.cublasCheckStatus(status)

    def cublasSgemm_v2(
        self,
        handle: int,
        transa: str,
        transb: str,
        m: int,
        n: int,
        k: int,
        alpha: int,
        A: int,
        lda: int,
        B: int,
        ldb: int,
        beta: int,
        C: int,
        ldc: int,
    ):
        status = self._libcublas.cublasSgemm_v2(
            handle,
            constant.CUBLAS_OP[transa],
            constant.CUBLAS_OP[transb],
            m,
            n,
            k,
            ctypes.byref(alpha),
            int(A),
            lda,
            int(B),
            ldb,
            ctypes.byref(beta),
            int(C),
            ldc,
        )
        self.cublasCheckStatus(status)

    def cublasHgemm(
        self,
        handle: int,
        transa: str,
        transb: str,
        m: int,
        n: int,
        k: int,
        alpha: int,
        A: int,
        lda: int,
        B: int,
        ldb: int,
        beta: int,
        C: int,
        ldc: int,
    ):
        status = self._libcublas.cublasHgemm(
            handle,
            constant.CUBLAS_OP[transa],
            constant.CUBLAS_OP[transb],
            m,
            n,
            k,
            ctypes.byref(alpha),
            int(A),
            lda,
            int(B),
            ldb,
            ctypes.byref(beta),
            int(C),
            ldc,
        )
        self.cublasCheckStatus(status)

    def cublasGemmEX(
        self,
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A,
        Atype,
        lda,
        B,
        Btype,
        ldb,
        beta,
        C,
        Ctype,
        ldc,
        computeType,
        algo,
    ):
        status = self._libcublas.cublasGemmEx(
            handle,
            constant.CUBLAS_OP[transa],
            constant.CUBLAS_OP[transb],
            m,
            n,
            k,
            ctypes.byref(alpha),
            int(A),
            constant.CUBLAS_DATA_TYPE[Atype],
            lda,
            int(B),
            constant.CUBLAS_DATA_TYPE[Btype],
            ldb,
            ctypes.byref(beta),
            int(C),
            constant.CUBLAS_DATA_TYPE[Ctype],
            ldc,
            constant.CUBLAS_COMPUTE_TYPE[computeType],
            constant.CUBLAS_GEMM_ALGO[algo],
        )
        self.cublasCheckStatus(status)
