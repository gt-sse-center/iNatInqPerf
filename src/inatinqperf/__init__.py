"""Package level __init__."""

from importlib.metadata import version

__version__ = version("inatinqperf")

from sys import platform

# On MacOS multiple linking of OpenMP happens and causes the program to segfault.
# Setting the KMP_DUPLICATE_LIB_OK environment variable seems to be the best workaround.
# Please see: https://github.com/dmlc/xgboost/issues/1715
if platform == "darwin":
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Disable multithreading on MacOS arm64 since it causes FAISS to crash.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
