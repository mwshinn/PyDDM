from paranoid import ParametersDict, Range, Natural0, Set

class Conditions(ParametersDict):
    """Valid conditions for our task."""
    def __init__(self):
        super().__init__({"coherence": Range(50, 100),
                          "presample": Natural0,
                          "highreward": Set([0, 1]),
                          "blocktype": Set([1, 2])})
    def test(self, v):
        super().test(v)
        assert isinstance(v, dict), "Non-dict passed"
        assert not set(v.keys()) - {'coherence', 'presample', 'highreward', 'blocktype'}, \
            "Invalid reward keys"
    def generate(self):
        for ps,coh,hr,bt in zip([0, 400, 800, 1000], [50, 53, 57, 60, 70], [0, 1], [1, 2]):
            yield {"presample": ps, "coherence": coh, "highreward": hr, "blocktype": bt}
            
