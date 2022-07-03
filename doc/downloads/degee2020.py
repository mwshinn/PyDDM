#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import pandas as pd

import pyddm as ddm
from pyddm import Sample
from pyddm import plot
from pyddm import models
from pyddm import Model, Fittable, Fitted, Bound, Overlay, Solution
from pyddm.functions import fit_adjust_model, display_model
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture, OverlayUniformMixture, InitialCondition, ICPoint, ICPointSourceCenter, LossBIC


# Start get_param_names
def get_param_names(sample, depends_on, param):

    if depends_on is None:
        unique_conditions = None
        names = [param]
    else:
        unique_conditions = [np.unique(np.concatenate((np.unique(sample.conditions[depends_on[i]][0]), 
                                        np.unique(sample.conditions[depends_on[i]][1])))) for i in range(len(depends_on))]
        if len(unique_conditions) == 1:
            names = ['{}{}'.format(param,i) for i in unique_conditions[0]]
        elif len(unique_conditions) == 2:
            names = ['{}{}.{}'.format(param,i,j) for i in unique_conditions[0] for j in unique_conditions[1]]

    return names, unique_conditions
# End get_param_names

# Start maker_functions
def make_z(sample, z_depends_on=[None]):
    
    z_names, z_unique_conditions = get_param_names(sample=sample, depends_on=z_depends_on, param='z')

    class StartingPoint(InitialCondition):
        name = 'A starting point.'
        required_parameters = z_names
        if not z_depends_on is None:
            required_conditions = z_depends_on.copy()
        def get_IC(self, x, dx, conditions):
            pdf = np.zeros(len(x))
            if z_depends_on is None:
                z_param = self.z
            elif len(z_unique_conditions) == 1:
                z_param = getattr(self, 'z{}'.format(conditions[z_depends_on[0]]))
            elif len(z_unique_conditions) == 2:
                z_param = getattr(self, 'z{}.{}'.format(conditions[z_depends_on[0]],conditions[z_depends_on[1]]))
            pdf[int(len(pdf)*z_param)] = 1
            return pdf
    return StartingPoint

def make_drift(sample, drift_bias, v_depends_on=[None], b_depends_on=[None]):
    
    v_names, v_unique_conditions = get_param_names(sample=sample, depends_on=v_depends_on, param='v')
    if drift_bias:
        b_names, b_unique_conditions = get_param_names(sample=sample, depends_on=b_depends_on, param='b')
    else:
        b_names = []

    class DriftStimulusCoding(ddm.models.Drift):
        name = 'Drift'
        required_parameters = v_names + b_names
        required_conditions = ['stimulus']
        if (v_depends_on is not None):
            required_conditions = list(set(required_conditions+v_depends_on))
        if (b_depends_on is not None):
            required_conditions = list(set(required_conditions+b_depends_on))

        def get_drift(self, conditions, **kwargs):
            
            # v param:
            if v_depends_on is None:
                v_param = self.v
            elif len(v_unique_conditions) == 1:
                v_param = getattr(self, 'v{}'.format(conditions[v_depends_on[0]]))
            elif len(v_unique_conditions) == 2:
                v_param = getattr(self, 'v{}.{}'.format(conditions[v_depends_on[0]],conditions[v_depends_on[1]]))

            if drift_bias:
                # b param:
                if b_depends_on is None:
                    b_param = self.b
                elif len(b_unique_conditions) == 1:
                    b_param = getattr(self, 'b{}'.format(conditions[b_depends_on[0]]))
                elif len(b_unique_conditions) == 2:
                    b_param = getattr(self, 'b{}.{}'.format(conditions[b_depends_on[0]],conditions[b_depends_on[1]]))
            
            # return:
                return b_param + (v_param * conditions['stimulus'])
            else:
                return (v_param * conditions['stimulus'])
    return DriftStimulusCoding

def make_a(sample, urgency, a_depends_on=[None], u_depends_on=[None]):
    
    a_names, a_unique_conditions = get_param_names(sample=sample, depends_on=a_depends_on, param='a')
    if urgency:
        u_names, u_unique_conditions = get_param_names(sample=sample, depends_on=u_depends_on, param='u') 
    else:
        u_names = []
    
    class BoundCollapsingHyperbolic(Bound):
        name = 'Hyperbolic collapsing bounds'
        required_parameters = a_names + u_names
        
        required_conditions = []
        if (a_depends_on is not None):
            required_conditions = list(set(required_conditions+a_depends_on))
        if (u_depends_on is not None):
            required_conditions = list(set(required_conditions+u_depends_on))
        
        def get_bound(self, t, conditions, **kwargs):
            
            # a param:
            if a_depends_on is None:
                a_param = self.a
            elif len(a_unique_conditions) == 1:
                a_param = getattr(self, 'a{}'.format(conditions[a_depends_on[0]]))
            elif len(a_unique_conditions) == 2:
                a_param = getattr(self, 'a{}.{}'.format(conditions[a_depends_on[0]],conditions[a_depends_on[1]]))
            
            if urgency:
                # u param:
                if u_depends_on is None:
                    u_param = self.u
                elif len(u_unique_conditions) == 1:
                    u_param = getattr(self, 'u{}'.format(conditions[u_depends_on[0]]))
                elif len(u_unique_conditions) == 2:
                    u_param = getattr(self, 'u{}.{}'.format(conditions[u_depends_on[0]],conditions[u_depends_on[1]]))
                
            # return:
                return a_param-(a_param*(t/(t+u_param)))
            else:
                return a_param
    return BoundCollapsingHyperbolic

def make_t(sample, t_depends_on=[None]):
    
    t_names, t_unique_conditions = get_param_names(sample=sample, depends_on=t_depends_on, param='t')
    
    class NonDecisionTime(Overlay):
        name = 'Non-decision time'
        required_parameters = t_names
        if not t_depends_on is None:
            required_conditions = t_depends_on.copy()
        def apply(self, solution):
            # Unpack solution object
            corr = solution.corr
            err = solution.err
            m = solution.model
            cond = solution.conditions
            undec = solution.undec
            
            # t param:
            if t_depends_on is None:
                t_param = self.t
            elif len(t_unique_conditions) == 1:
                t_param = getattr(self, 't{}'.format(cond[t_depends_on[0]]))
            elif len(t_unique_conditions) == 2:
                t_param = getattr(self, 't{}.{}'.format(cond[t_depends_on[0]],cond[t_depends_on[1]]))
            
            shifts = int(t_param/m.dt) # truncate
            # Shift the distribution
            newcorr = np.zeros(corr.shape, dtype=corr.dtype)
            newerr = np.zeros(err.shape, dtype=err.dtype)
            if shifts > 0:
                newcorr[shifts:] = corr[:-shifts]
                newerr[shifts:] = err[:-shifts]
            elif shifts < 0:
                newcorr[:shifts] = corr[-shifts:]
                newerr[:shifts] = err[-shifts:]
            else:
                newcorr = corr
                newerr = err
            return Solution(newcorr, newerr, m, cond, undec)
    return NonDecisionTime
# End maker_functions

# Start make_model
def make_model(sample, model_settings):
    
    # model components:
    z = make_z(sample=sample, 
                z_depends_on=model_settings['depends_on']['z'])
    drift = make_drift(sample=sample, 
                        drift_bias=model_settings['drift_bias'], 
                        v_depends_on=model_settings['depends_on']['v'], 
                        b_depends_on=model_settings['depends_on']['b'])
    a = make_a(sample=sample, 
                urgency=model_settings['urgency'], 
                a_depends_on=model_settings['depends_on']['a'], 
                u_depends_on=model_settings['depends_on']['u'])
    t = make_t(sample=sample, 
                t_depends_on=model_settings['depends_on']['t'])
    T_dur = model_settings['T_dur']

    # limits:
    ranges = {
            'z':(0.05,0.95),               # starting point
            'v':(0,5),                     # drift rate
            'b':(-5,5),                    # drift bias
            'a':(0.1,5),                   # bound
            # 'u':(-T_dur*10,T_dur*10),    # hyperbolic collapse
            'u':(0.01,T_dur*10),           # hyperbolic collapse
            't':(0,2),                     # non-decision time
            }

    # put together:
    if model_settings['start_bias']:
        initial_condition = z(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in z.required_parameters})
    else:
        initial_condition = z(**{'z':0.5})
    model = Model(name='stimulus coding model / collapsing bound',
                IC=initial_condition,
                drift=drift(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in drift.required_parameters}),
                bound=a(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in a.required_parameters}),
                overlay=OverlayChain(overlays=[t(**{param:Fittable(minval=ranges[param[0]][0], maxval=ranges[param[0]][1]) for param in t.required_parameters}),
                                                OverlayUniformMixture(umixturecoef=0)]),
                                                # OverlayPoissonMixture(pmixturecoef=.01, rate=1)]),
                noise=NoiseConstant(noise=1),
                dx=.005, dt=.01, T_dur=T_dur)
    return model

# End make_model
