import torch
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()

x = torch.randn(1, 3, 224, 224)

output = resnet18(x)
print(output.shape)

torch.jit.trace(resnet18, x).save("model_file/resnet18.pt")