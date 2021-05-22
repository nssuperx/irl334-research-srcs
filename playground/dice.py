import numpy as np
from numpy.core.fromnumeric import size

def main():
    trialNum = 1000
    diceArray = np.random.randint(1, 7, size=trialNum)
    diceMean = diceArray.mean()
    print(diceMean)
    

if __name__ == "__main__":
    main()
