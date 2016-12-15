from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from DDM_model import *
from DDM_parameters import *


mu_0 = 0
sigma_0 = 1
B = 1
s1 = Model(name="DDM", mu=mu_0, sigma=sigma_0, B=B,
           mudep=MuLinear(x=0, t=0),
           sigmadep=SigmaLinear(x=0, t=0),
           bounddep=BoundConstant())
s2 = Model(name="CB_Lin", mu=mu_0, sigma=sigma_0, B=B,
           mudep=MuLinear(x=0, t=0),
           sigmadep=SigmaLinear(x=0, t=0),
           bounddep=BoundCollapsingLinear(t=.01))
s3 = Model(name="CB_Expo", mu=mu_0, sigma=sigma_0, B=B,
           mudep=MuLinear(x=0, t=0),
           sigmadep=SigmaLinear(x=0, t=0),
           bounddep=BoundCollapsingExponential(tau=1))
s4 = Model(name="OU+", mu=mu_0, sigma=sigma_0, B=B,
           mudep=MuLinear(x=0, t=0),
           sigmadep=SigmaLinear(x=0, t=0),
           bounddep=BoundConstant())
s5 = Model(name="OU-", mu=mu_0, sigma=sigma_0, B=B,
           mudep=MuLinear(x=0, t=0),
           sigmadep=SigmaLinear(x=0, t=0),
           bounddep=BoundConstant())
models = [s1, s2, s3, s4, s5]

