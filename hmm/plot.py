import matplotlib.pyplot as plt
import numpy as np

def main():
    loaddata = np.loadtxt('out.csv', delimiter=',')
    loaddata = loaddata.T
    time = np.array(loaddata[0], dtype=np.int8)
    x = np.array(loaddata[1], dtype=np.int8)
    y = np.array(loaddata[2], dtype=np.float64)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(time, x)
    ax.plot(time, y, linestyle = 'None', marker='.')
    plt.show()

if __name__ == "__main__":
    main()
