import matplotlib.pyplot as plt
from pyddm import Model
m = Model()
s = m.solve()
plt.plot(s.model.t_domain(), s.pdf("correct"))
plt.savefig("helloworld.png")
plt.show()
