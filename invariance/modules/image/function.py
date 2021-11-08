import numpy as np

def scan(originalImgArray: np.ndarray, templateImgArray: np.ndarray) -> np.ndarray:
    # 走査
    # TODO: 遅すぎるのでなんとかする
    scanImgArray = np.empty((originalImgArray.shape[0] - templateImgArray.shape[0], originalImgArray.shape[1] - templateImgArray.shape[1]))
    for i in range(scanImgArray.shape[0]):
        for j in range(scanImgArray.shape[1]):
            # 相関係数出す範囲をスライス
            scanTargetImgArray = originalImgArray[i:i+templateImgArray.shape[0], j:j+templateImgArray.shape[1]]
            cov = np.mean(np.multiply(scanTargetImgArray, templateImgArray))
            # scanImgArray[i][j] = np.corrcoef(scanTargetImgArray.flatten(), templateImgArray.flatten())[0][1]
            scanImgArray[i][j] = cov / (scanTargetImgArray.std() * templateImgArray.std())

    return scanImgArray
