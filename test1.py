import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torch.autograd import Variable
import os
classes = ('Real', 'AI-generated')
transform_test = transforms.Compose([
    transforms.CenterCrop(600),  # EfficientNet-B7推荐尺寸
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth")
model.eval()
model.to(DEVICE)

path='data/test/'
testList=os.listdir(path)
for file in testList:
    img=Image.open(path+file)
    img=transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out=model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file,classes[pred.data.item()]))