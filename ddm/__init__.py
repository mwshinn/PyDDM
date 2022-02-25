# This file is to provide backward compatibility.
import warnings
warnings.warn(
    "PyDDM should be imported using the 'pyddm' module instead of the 'ddm' module.  "
    "Please use 'import pyddm' or 'import pyddm as ddm' instead of 'import ddm'.  "
    "The 'ddm' alias is deprecated and will be removed in future versions.",
    FutureWarning, stacklevel=20)
del warnings

# Make everything from pyddm available here.  Not sure if this is necessary
# given the assignment in sys.modules, but it never hurts to be safe.
import pyddm
import pyddm.models as models
__name__ = "pyddm"
globals().update(pyddm.__dict__)

# We must set each of these manually.  Otherwise, when you use "from ... import
# ..." syntax, you get an identical object with the wrong ID, which causes
# problems when checking with "isinstance".  For some reason this doesn't
# happen with a plain "import".  It must be some bug or quirk with the import
# system.
sys.modules['ddm'] = sys.modules['pyddm']
sys.modules['ddm.models'] = sys.modules['pyddm.models']
sys.modules['ddm.models.drift'] = sys.modules['pyddm.models.drift']
sys.modules['ddm.models.noise'] = sys.modules['pyddm.models.noise']
sys.modules['ddm.models.ic'] = sys.modules['pyddm.models.ic']
sys.modules['ddm.models.base'] = sys.modules['pyddm.models.base']
sys.modules['ddm.models.overlay'] = sys.modules['pyddm.models.overlay']
sys.modules['ddm.models.bound'] = sys.modules['pyddm.models.bound']
sys.modules['ddm.models.loss'] = sys.modules['pyddm.models.loss']
sys.modules['ddm.functions'] = sys.modules['pyddm.functions']
sys.modules['ddm.solution'] = sys.modules['pyddm.solution']
sys.modules['ddm.sample'] = sys.modules['pyddm.sample']
sys.modules['ddm.tridiag'] = sys.modules['pyddm.tridiag']
sys.modules['ddm.fitresult'] = sys.modules['pyddm.fitresult']
sys.modules['ddm.analytic'] = sys.modules['pyddm.analytic']
