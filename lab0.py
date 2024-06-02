import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
#plt.plot(x, x + 0, linestyle='solid')
#plt.plot(x, x + 1, linestyle='dashed')
#plt.plot(x, x + 2, linestyle='dashdot')
#plt.plot(x, x + 3, linestyle='dotted')
plt.plot(x, x + 0, '-g')  # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r');  # dotted red
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':');  # dotted
plt.xlabel("x")
plt.ylabel("y")
plt.title("A sine curve")
plt.show()