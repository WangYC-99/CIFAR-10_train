import torch 
import torchvision 
from torch import nn
from torchvision import transforms
from PIL import Image 
from torch.utils.tensorboard import SummaryWriter

img_path = "images/airplane.png"
image = Image.open(img_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
image = torch.reshape(image, (1, 3, 32, 32))

class Module_WangYC(nn.Module):
    def __init__(self):
        super(Module_WangYC, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

module = torch.load("saved_modules/WangYC_Module.pth", map_location=torch.device('cpu'))
module.eval()
with torch.no_grad():
    output = module(image)

result = output.argmax(1).item()

if result == 0:
    print("airplane")
elif result == 1:
    print("automobile")
elif result == 2:
    print("bird")
elif result == 3:
    print("cat")
elif result == 4:
    print("deer")
elif result == 5:
    print("dog")
elif result == 6:
    print("frog")
elif result == 7:
    print("horse")
elif result == 8:
    print("ship")
elif result == 9:
    print("truck")
