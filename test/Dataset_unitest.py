from torchvision.datasets import ImageFolder

dataroot = '/opt/ml/input/purified/train/'
mydata = ImageFolder(dataroot)

if __name__ == '__main__':
    prev = None
    for data, cls in mydata.samples:
        curr = cls
        if prev != curr:
            print(f'{curr}\t: {data}')
            prev = curr
