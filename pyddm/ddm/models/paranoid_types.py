from paranoid import Dict, String, Number
class Conditions(Dict):
    """Valid conditions for a model"""
    def __init__(self):
        super().__init__(String, Number)
