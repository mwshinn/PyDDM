from paranoid import Dict, String, Number, Or, Unchecked
class Conditions(Dict):
    """Valid conditions for a model"""
    def __init__(self):
        super().__init__(String, Unchecked(Or(Number, String)))
