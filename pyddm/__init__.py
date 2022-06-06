# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

__all__ = ["Sample", "Solution", "Model", "Fittable", "Fitted", "FitResult"]

# Check that Python3 is running
import sys
if sys.version_info.major != 3:
    raise ImportError("PyDDM only supports Python 3")

from .models import *
from .model import Model, Fittable, Fitted, FitResult
from .sample import Sample
from .solution import Solution
from .functions import *
from .logger import set_log_level

from ._version import __version__

# Some default functions for paranoid scientist
import paranoid
import math
import numpy as np
paranoid.settings.Settings.get("namespace").update({"math": math, "np": np})
# Disable paranoid for users
paranoid.settings.Settings.set(enabled=False)
