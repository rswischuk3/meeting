import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

with open('tas', 'r') as fp:
	tas = pickle.load(fp)

"""
=========================
Simple animation examples
=========================

This example contains two animations. The first is a random walk plot. The
second is an image animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
print np.arange(5)

def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

fig1 = plt.figure()

data = np.zeros((2,tas[0].shape[0]))
data[0,:] = np.arange(tas[0].shape[0])
data[1,:] = tas[0]
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, 100, fargs=(data, l),
                                   interval=200, blit=True)
								   
plt.show()