import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load('./hmm_results.npz')
    x = data['x']
    y = data['y']
    z = data['z']

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(x,y,z)
    plt.show()

if __name__ == '__main__':
    main()
