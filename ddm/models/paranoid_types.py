# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

from paranoid import Dict, String, Number, Or, Unchecked
class Conditions(Dict):
    """Valid conditions for a model"""
    def __init__(self):
        super().__init__(String, Unchecked(Or(Number, String)))
