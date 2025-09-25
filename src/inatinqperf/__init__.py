"""Package level __init__."""

from importlib.metadata import version

__version__ = version("inatinqperf")

# On MacOS multiple linking of OpenMP happens and causes the program to segfault.
# Setting the KMP_DUPLICATE_LIB_OK environment variable seems to be the best workaround.
# Please see: https://github.com/dmlc/xgboost/issues/1715

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
