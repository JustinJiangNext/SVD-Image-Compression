import cv2, os
import numpy as np
import pickle, lzma

class CompressedImage:
    def __init__(self, UMatrices: list[np.ndarray], SMatrices: list[np.ndarray], VTMatrices: list[np.ndarray]):
        self.UMatrices: list[np.ndarray] = UMatrices
        self.SMatrices: list[np.ndarray] = SMatrices
        self.VTMatrices: list[np.ndarray] = VTMatrices
        self.num_channels:int = len(UMatrices)

    @classmethod
    def loadImage(self, image: np.ndarray):
        UMatrices: list[np.ndarray] = []
        SMatrices: list[np.ndarray] = []
        VTMatrices: list[np.ndarray] = []
        num_channels:int = image.shape[-1:][0]
        for color_channel in range(num_channels):
            U, s, Vt = np.linalg.svd(image[:,:,color_channel], full_matrices=False)
            UMatrices.append(U)
            SMatrices.append(s)
            VTMatrices.append(Vt)
        return CompressedImage(UMatrices, SMatrices, VTMatrices)

    @classmethod
    def loadImageFile(self, path: str, code: int = -1):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if code != -1:
            img = cv2.cvtColor(img, code)
        return CompressedImage.loadImage(img)
    
    @classmethod
    def loadCompressedImage(self, path: str,):
        with lzma.open(path, "rb") as dbfile:
            db = pickle.load(dbfile)
        return CompressedImage(db["UMatrices"], db["SMatrices"], db["VTMatrices"])

    def kthLayer(self, k:int) -> np.ndarray:
        channels: list[np.ndarray] = []
        for color_channel in range(self.num_channels):
            channels.append(self.SMatrices[color_channel][k] * np.outer(self.UMatrices[color_channel][:,k], self.VTMatrices[color_channel][k,:]))
        return np.stack(channels, axis=2)

    def kLayerApproximation(self, k:int, sanatize = True) -> np.ndarray:
        approximation: np.ndarray = np.zeros((self.UMatrices[0].shape[0], self.VTMatrices[0].shape[1], self.num_channels), dtype=float)
        for layerK in range(k):
            approximation += self.kthLayer(layerK)
        if sanatize:
            return np.clip(approximation, 0, 255).astype(np.uint8)
        return approximation
    
    def kLayerApproximationFast(self, k:int, sanatize = True) -> np.ndarray:
        channels: list[np.ndarray] = []
        for color_channel in range(self.num_channels):
            channels.append((self.UMatrices[color_channel][:,:k] * self.SMatrices[color_channel][:k][np.newaxis, :]) @ self.VTMatrices[color_channel][:k,:])
        approximation =  np.stack(channels, axis=2)
        if sanatize:
            return np.clip(approximation, 0, 255).astype(np.uint8)
        return approximation
    
    def saveImage(self, path:str, maxK: int) -> None:
        db = {}
        if(maxK == -1):
            maxK = min(self.UMatrices[0].shape[1], self.VTMatrices[0].shape[0])
        db['UMatrices'] = [arr[:, :maxK] for arr in self.UMatrices]
        db['SMatrices'] = [arr[:maxK] for arr in self.SMatrices]
        db['VTMatrices'] = [arr[:maxK,:]for arr in self.VTMatrices]
        with lzma.open(path, "wb") as dbfile:
            pickle.dump(db, dbfile)                    
        dbfile.close()
        return os.path.getsize(path)

    def maxApproximation(self) -> np.ndarray:
        return self.kLayerApproximation(self.getMaxK())
    
    def getMaxK(self) -> int:
        return min(self.UMatrices[0].shape[1], self.VTMatrices[0].shape[0])
    
    def getDim(self) -> tuple[int, int]:
        return (self.UMatrices[0].shape[0], self.VTMatrices[0].shape[1])
    
    def getNumBytes(self, k = -1) -> int:
        if k == -1:
            k = self.getMaxK()
        numBytes:int = 0
        for color_channel in range(self.num_channels):
            numBytes += self.UMatrices[color_channel][:,:k].nbytes
            numBytes += self.VTMatrices[color_channel][:k,:].nbytes
            numBytes += self.SMatrices[color_channel][:k].nbytes
        return numBytes
    
    def findK(self, accuracy: int) -> int:
        accuracy = int(np.clip(accuracy, 0, 99))
        target_rel_err = 1.0 - (accuracy / 100.0)   # want avg_rel_err <= this

        # maxK based on available singular values (safe)
        maxK = min(len(s) for s in self.SMatrices)
        if maxK < 1:
            return 0  # nothing to do / degenerate

        C = self.num_channels

        # s2: (C, maxK)
        s2 = np.stack([self.SMatrices[c][:maxK].astype(np.float64) ** 2
                        for c in range(C)], axis=0)

        tot2 = s2.sum(axis=1)  # (C,)
        # avoid divide-by-zero: if a channel is all-zero, define rel_err=0 for all k
        nonzero = tot2 > 0.0

        # tail2[:, k-1] = sum_{i>k} s_i^2
        prefix2 = np.cumsum(s2, axis=1)              # (C, maxK)
        tail2 = tot2[:, None] - prefix2              # (C, maxK)

        rel_err = np.zeros((C, maxK), dtype=np.float64)
        rel_err[nonzero] = np.sqrt(tail2[nonzero] / tot2[nonzero, None])

        avg_rel_err = rel_err.mean(axis=0)           # (maxK,) for k=1..maxK

        hits = np.flatnonzero(avg_rel_err <= target_rel_err)
        return int(hits[0] + 1) if hits.size else maxK


