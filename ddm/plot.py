import pyddm.plot
__name__ = "pyddm.plot"

globals().update(pyddm.plot.__dict__)

sys.modules['ddm.plot'] = sys.modules['pyddm.plot']
