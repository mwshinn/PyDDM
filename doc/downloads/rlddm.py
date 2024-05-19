# RL-DDM using PyDDM


# BEGIN SIMULATION_CODE
import pyddm
import pandas
import numpy as np
m_sim = pyddm.gddm(drift=lambda deltaq, driftscale : driftscale * deltaq,
                   noise=1,
                   bound=1,
                   nondecision=0,
                   parameters={"driftscale": 1},
                   conditions=["deltaq"], T_dur=10)

def sim_rlddm(n_trials, n_sessions, reward_probabilities, alpha):
    qvals = np.array([0.5, 0.5])
    choice = []
    reward = []
    trial = []
    session = []
    rt = []
    qleft = []
    qright = []
    for sess in range(0, n_sessions):
        for t in range(0, n_trials):
            # compute choice probabilities using softmax formula
            sol = m_sim.solve(conditions={"deltaq": qvals[1]-qvals[0]})
            res = sol.sample(5).to_pandas_dataframe(drop_undecided=True) # We only use the first non-undecided trial
            r = np.random.randint(0, 5)
            choice.append(res['choice'][r]) # 1 for right and 0 for left
            rt.append(float(res['RT'][r]))
            session.append(sess)
            trial.append(t)
            reward.append(int(np.random.uniform() < reward_probabilities[choice[-1]]))
            qvals[choice[-1]] = (1-alpha)*qvals[choice[-1]] + alpha*reward[-1]
            qleft.append(qvals[0])
            qright.append(qvals[1])
    df = pandas.DataFrame({"trial": trial, "session": session, "choice": choice, "reward": reward, "rt": rt})
    return pyddm.Sample.from_pandas_dataframe(df, choice_column_name="choice", rt_column_name="rt")

samp = sim_rlddm(1000, 2, [.2, .8], .1)
# END SIMULATION_CODE

# BEGIN LOSS
class LossRL(pyddm.LossFunction):
    name = "rl_loss"
    def setup(self, **kwargs):
         self.sessions = self.sample.condition_values('session')
         for s in self.sessions:
             trials = self.sample.subset(session=s).condition_values('trial')
             assert set(trials) == set(range(min(trials), min(trials)+len(trials))), "Trials must be sequentially numbered"
         self.df =  self.sample.to_pandas_dataframe()
         self.sessdfs = [self.df.query(f'session == {s}').sort_values('trial') for s in self.sessions]
    def loss(self, model):
        likelihood = 0
        qleft = [.5]
        qright = [.5]
        alpha = model.get_dependence("drift").alpha
        for j in range(0, len(self.sessions)):
            sessdf = self.sessdfs[j]
            for i,row in sessdf.iterrows():
                chose_left = row['choice'] <= 0
                p = model.solve_analytical(conditions={"deltaq": qright[-1]-qleft[-1]}).evaluate(row['RT'], not chose_left)
                if chose_left:
                    qleft.append((1-alpha) * qleft[-1] + alpha * row['reward'])
                    qright.append(qright[-1])
                else: # Right choice
                    qright.append((1-alpha) * qright[-1] + alpha * row['reward'])
                    qleft.append(qleft[-1])
                if p <= 0:
                    return -np.inf
                likelihood += np.log(p)
        model.last_qleft = qleft
        model.last_qright = qright
        return -likelihood
# END LOSS

# BEGIN FASTLOSS
class LossRLFast(pyddm.LossFunction):
    name = "rl_loss"
    ROUND = 2 # Number of decimal digits to round to, lower number gives better performance but lower accuracy
    def setup(self, **kwargs):
         self.sessions = self.sample.condition_values('session')
         for s in self.sessions:
             trials = self.sample.subset(session=s).condition_values('trial')
             assert set(trials) == set(range(min(trials), min(trials)+len(trials))), "Trials must be sequentially numbered"
         self.df =  self.sample.to_pandas_dataframe().sort_values(["session", "trial"]).reset_index()
         self.df['deltaq'] = 0
         self.sessinds = [self.df['session'] == s for s in self.sessions]
    def loss(self, model):
        likelihood = 0
        alpha = model.get_dependence("drift").alpha
        for j in range(0, len(self.sessions)):
            qleft = [.5]
            qright = [.5]
            for _,row in self.df[self.sessinds[j]].iterrows():
                chose_left = row['choice'] <= 0
                if chose_left:
                    qleft.append((1-alpha) * qleft[-1] + alpha * row['reward'])
                    qright.append(qright[-1])
                else: # Right choice
                    qright.append((1-alpha) * qright[-1] + alpha * row['reward'])
                    qleft.append(qleft[-1])
            self.df.loc[self.sessinds[j],'deltaq'] = np.round(np.asarray(qright)[:-1] - np.asarray(qleft)[:-1], self.ROUND)
        likelihood = pyddm.get_model_loss(model=model, sample=pyddm.Sample.from_pandas_dataframe(self.df, 'RT', 'choice'), lossfunction=pyddm.LossLikelihood)
        model.last_qleft = np.asarray(qleft)[:-1]
        model.last_qright = np.asarray(qright)[:-1]
        return likelihood
# END FASTLOSS

# BEGIN MODEL
m = pyddm.gddm(drift=lambda driftscale,deltaq,alpha : driftscale * deltaq, # Hack including alpha
               noise=1,
               bound=1,
               nondecision=0,
               T_dur=20, dx=.01, dt=.01,
               conditions=["deltaq", "session", "trial", "reward"],
               parameters={"driftscale": (0, 8), "alpha": (0, 1)})

# Fit the model
m.fit(sample=samp, lossfunction=LossRLFast)

# To get the qleft and qright values for the last session, run this:
pyddm.get_model_loss(model=m, sample=samp, lossfunction=LossRLFast)
qleft = m.last_qleft
qright = m.last_qright

pyddm.plot.model_gui(m, conditions={"deltaq": [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]})
# END MODEL
