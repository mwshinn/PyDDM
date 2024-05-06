import matplotlib.pyplot as plt
import pyddm
m = pyddm.gddm()
s = m.solve()
plt.plot(s.t_domain, s.pdf("correct"))
plt.savefig("helloworld.png")
plt.show()
