import ddm

class DriftCond(ddm.Drift):
    name = "Drift with a condition"
    required_conditions = ['cond']
    required_parameters = ["param"]
    def get_drift(self, conditions, **kwargs):
        return conditions["cond"]*self.param
