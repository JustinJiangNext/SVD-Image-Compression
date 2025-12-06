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