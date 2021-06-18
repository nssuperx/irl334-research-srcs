from matplotlib import scale
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

"""
x = np.arange(-3, 3, 0.1)
y = x**2 + 3
plt.plot(x, y)
plt.ylim(0, 15)
plt.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)
plt.tick_params(bottom=False,
               left=False,
               right=False,
               top=False)
plt.show()
"""

line = np.linspace(0, 20, 100)
locList = [5, 7.5, 10, 12.5, 15]
locList = [12.5]
for x in locList:
    kai2 = norm.pdf(line, loc=x, scale=0.5)
    plt.plot(range(100), kai2, color="#1f77b4")

# plt.xlim(0, 20)
plt.show()
