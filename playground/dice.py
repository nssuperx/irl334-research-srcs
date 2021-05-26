import numpy as np

def main():
    N = 400                # 1回の試行で投げるサイコロの数
    M = 50                 # めったに起こらない場合のときの配列をいくつ集めるか

    batchSize = 1000        # 何回分まとめて実験するか．大きくすると速度が上がるが，メモリをたくさん使う．

    miracleArray = np.empty((M, N))
    meanList = []

    miracleNum = 0
    while miracleNum < M:
        diceArray = np.random.randint(1, 7, size=(batchSize, N))
        diceMean = np.mean(diceArray, axis=1)
        
        # np.whereで複数の条件を適用したい場合，& や | を使う．
        # ndarrayのtupleが返ってくるので，[0]を指定
        for diceArrayIndex in np.where((3.1 < diceMean) & (diceMean < 3.2))[0]:
            miracleArray[miracleNum] = np.copy(diceArray[diceArrayIndex])
            meanList.append(diceMean[diceArrayIndex])
            miracleNum += 1
            if(miracleNum >= M):
                break

    # 出現回数カウント用numpy配列
    countArray = np.empty((M, 6))
    for i in range(6):
        countArray[:, i] = np.count_nonzero(miracleArray == (i+1), axis=1)

    countArray = countArray / N

    for i in range(M):
        print(str(countArray[i]) + " : " + str(meanList[i]))


if __name__ == "__main__":
    main()



"""
メモ

1,2,3,4,5,6
0.1, 0.1, 0.2, 0.1, 0.2, 0.3
平均値4.1, X^2の平均値 = 19.7

3.6 < 平均 < 3.8
このとき，X^2の平均値 < 17


"""
