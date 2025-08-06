import torch
import torchvision
import torchvision.transforms as transforms

from myresnet import ResidualBlock, ResNet

# load dataset and transform it for training
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.4465], std=[0.202, 0.1994, 0.2010]),
    ]
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transform, download=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=True, num_workers=2
)

# set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def accuracy(model):
    model.eval()  # ↩️ disables Dropout/BatchNorm updates
    correct = total = 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct * 100 / total


for tag, ckpt_path in [
    ("resnet-20", "./data/resnet20_cifar10.pth"),
    ("resent-32", "./data/resnet32_cifar10.pth"),
]:
    n = (3 if tag == "resnet-20" else 5) #following the 6n+2 rule

    model = ResNet(ResidualBlock, n).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    acc = accuracy(model)
    print(f"{tag:7s}: {acc:.2f}% top-1 accuracy")
