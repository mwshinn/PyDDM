from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from DDM_model import *
from DDM_parameters import *


mu_0 = 0
sigma_0 = 1
B = 1
s1 = Model(name="DDM", 
           mu=MuConstant(mu=mu_0),
           sigma=SigmaConstant(sigma=sigma_0),
           bound=BoundConstant(B=B))
s2 = Model(name="CB_Lin",
           mu=MuConstant(mu=mu_0),
           sigma=SigmaConstant(sigma=sigma_0),
           bound=BoundCollapsingLinear(B=B, t=.01))
s3 = Model(name="CB_Expo",
           mu=MuConstant(mu=mu_0),
           sigma=SigmaConstant(sigma=sigma_0),
           bound=BoundCollapsingExponential(B=B, tau=1))
s4 = Model(name="OU+",
           mu=MuConstant(mu=mu_0),
           sigma=SigmaConstant(sigma=sigma_0),
           bound=BoundConstant(B=B))
s5 = Model(name="OU-",
           mu=MuConstant(mu=mu_0),
           sigma=SigmaConstant(sigma=sigma_0),
           bound=BoundConstant(B=B))
models = [s1, s2, s3, s4, s5]

