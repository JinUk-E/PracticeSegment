from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from unet import UNet
from BSDDataLoader import GetDataSet


class options:
    def __init__(self):
        self.cuda = True
        self.batchSize = 4
        self.testBatchSize = 4
        self.nEpochs = 140
        self.lr = 0.001
        self.threads = 4
        self.seed = 123
        self.size = 428
        self.remSize = 20
        self.colorDim = 1
        self.targetMode = 'bon'
        self.pretrainNet = "../checkpoint/model_epoch_140.pth"


def map01(tensor, eps=1e-5):
    max = np.max(tensor.numpy(), axis=(1, 2, 3), keepdims=True)
    min = np.min(tensor.numpy(), axis=(1, 2, 3), keepdims=True)
    if (max - min).any():
        return torch.from_numpy((tensor.numpy() - min) / (max - min + eps))
    else:
        return torch.from_numpy((tensor.numpy() - min) / (max - min))

def SizeIsValid(size):
    for i in range(4):
        size -= 4
        if size%2: return 0
        else: size /= 2

    for i in range(4):
        size -= 4
        size *= 2

    return size-4


opt = options()
targetSize = SizeIsValid(opt.size)
print("output size: ", targetSize)
if not targetSize:
    raise ValueError("Invalid size")

targetGap = (opt.size - targetSize) // 2
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
trainSet = GetDataSet(opt.size + opt.remSize, 'train', opt.targetMode, opt.colorDim)
testSet = GetDataSet(opt.size, 'test', opt.targetMode, opt.colorDim)
trainingDataLoader = DataLoader(dataset=trainSet, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testingDataLoader = DataLoader(dataset=testSet, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
unet = UNet(opt.colorDim)

criterion = nn.MSELoss()
if cuda:
    unet = unet.cuda()
    criterion = criterion.cuda()

preTrained = True
if preTrained:
    unet.load_state_dict(torch.load(opt.pretrainNet))

optimizer = optim.SGD(unet.parameters(), lr=opt.lr)
print('===> Training')

def train(epoch):
    epochLoss = 0

    for iteration, batch in enumerate(trainingDataLoader, 1):
        randH = random.randint(0, opt.remSize)
        randW = random.randint(0, opt.remSize)
        input = Variable(batch[0][:,:,randH:randH+opt.size, randW:randW+opt.size])
        target = Variable(batch[1][:,:,randH+targetGap:randH+targetGap+targetSize,
                          randW+targetGap:randW+targetGap+targetSize])

        if cuda:
            input = input.cuda()
            target = target.cuda()

        input = unet(input)
        loss = criterion(input, target)
        epochLoss += loss.data[0]
        loss.backward()
        optimizer.step()
        if iteration%10 is 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(trainingDataLoader), loss.data[0]))

    imgOut = input.data/2 + 1
    torchvision.utils.save_image(imgOut, "../output/epoch{}.png".format(epoch))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epochLoss / len(trainingDataLoader)))


def test():
    totalLoss = 0
    for batch in testingDataLoader:
        input = Variable(batch[0],volatile=True)
        target = Variable(batch[1][:, :,
                          targetGap:targetGap + targetSize,
                          targetGap:targetGap + targetSize],
                          volatile=True)

        if cuda:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        prediction = unet(input)
        loss = criterion(prediction, target)
        totalLoss += loss.data[0]
    print("===> Avg. Loss: {:.4f}".format(totalLoss / len(testingDataLoader)))

def checkpoint(epoch):
    modelOutPath = "../checkpoint/model_epoch_{}.pth".format(epoch)
    torch.save(unet.state_dict(), modelOutPath)
    print("Checkpoint saved to {}".format(modelOutPath))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    if epoch%10 is 0:
        checkpoint(epoch)
    test()
checkpoint(epoch)