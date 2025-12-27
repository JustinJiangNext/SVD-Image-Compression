import sys, os
from CompressedImage import CompressedImage 
import cv2

def main():
    try:
        args = sys.argv[1:]

        flagCompress = False
        flagAccuracy = False
        accuracyExpected = 85

        used = [False] * len(args)

        for i, arg in enumerate(args):
            if arg in ("-C", "-c", "--compress"):
                flagCompress = True
                used[i] = True
            elif arg in ("-a", "-A", "--accuracy"):
                flagAccuracy = True
                used[i] = True
                if i + 1 >= len(args):
                    raise ValueError
                acc = int(args[i + 1])
                if not (0 <= acc <= 99):
                    raise ValueError
                accuracyExpected = acc
                used[i + 1] = True

        names = []
        for i, arg in enumerate(args):
            if used[i]:
                continue
            if arg.isdigit():
                continue
            if arg.startswith("-"):
                raise ValueError
            names.append(arg)

        inputName = names[0] if len(names) >= 1 else ""
        outputName = names[1] if len(names) >= 2 else ""

        if(len(inputName) == 0):
            print("Command Line Error")
            sys.exit(1)
        
        if outputName == "":
            filename, _ = os.path.splitext(inputName)
            if(flagCompress):
                outputName = filename + ".svd"
            else:
                outputName = filename + ".png"

        if flagCompress: 
            cimg = CompressedImage.loadImageFile(inputName)
            cimg.saveImage(outputName, cimg.findK(accuracyExpected))
        else:
            cimg = CompressedImage.loadCompressedImage(inputName)
            cv2.imwrite(outputName, cimg.kLayerApproximationFast(cimg.findK(accuracyExpected)))
    except Exception as e:
        print("Command Line Error")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
