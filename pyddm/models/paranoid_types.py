# Copyright 2018 Max Shinn <maxwell.shinn@yale.edu>
#           2018 Norman Lam <norman.lam@yale.edu>
# 
# This file is part of PyDDM, and is available under the MIT license.
# Please see LICENSE.txt in the root directory for more information.

from paranoid import Dict, String, Identifier, Number, Or, Unchecked, Type, Set
class Conditions(Dict):
    """Valid conditions for a model"""
    def __init__(self):
        super().__init__(Identifier, Unchecked(Or(Number, String)))

class Choice(Type):
    """True or False"""
    def test(self, v):
        assert v in Or(String, Set([0, 1, 2]))
    def generate(self):
        yield "_top"
        yield "_bottom"
        yield 0
        yield 1
        yield 2
