from os.path import exists, join
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import torch.utils.data as data
from os import listdir
from PIL import Image


def SimpleRoot(dest="/dir/to/dataset"):
    if not exists(dest): raise ValueError("Path does not exist")
    return dest

def InputTransform(cropSize):
    return Compose([ Resize(cropSize), CenterCrop(cropSize), ToTensor() ])

def GetDataSet(size, folderName, targetMode='seg', colorDim=1):
    rootDir = SimpleRoot()
    targetDir = join(rootDir, folderName)
    return DataSetFromFolder(targetDir, targetMode, colorDim,
                            inputTransform=InputTransform(size),
                            targetTransform=InputTransform(size))

def IsImageFile(fileName):
    return any(fileName.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def LoadImage(filePath, colorDim):
    convertMode = 'L' if colorDim == 1 else 'RGB'
    return Image.open(filePath).convert(convertMode)

class DataSetFromFolder(data.Dataset):
    def __init__(self, folderPath, targetMode, colorDim, inputTransform=None, targetTransform=None):
        super(DataSetFromFolder, self).__init__()
        self.imagePaths = [join(folderPath, x) for x in listdir(folderPath) if IsImageFile(x)]
        self.targetMode = targetMode
        self.colorDim = colorDim
        self.inputTransform = inputTransform
        self.targetTransform = targetTransform

    def __getitem__(self, index):
        input = LoadImage(self.imagePaths[index], self.colorDim)
        target = LoadImage(self.imagePaths[index].replace('input', self.targetMode), self.colorDim)
        if self.inputTransform:
            input = self.inputTransform(input)
        if self.targetTransform:
            target = self.targetTransform(target)
        return input, target

    def __len__(self):
        return len(self.imagePaths)