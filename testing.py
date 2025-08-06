import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from myresnet import ResidualBlock, ResNet

# load dataset and transform it for training
train_transform = transforms.Compose(
    [
        transforms.Pad(4),  # pad 4 px on each side  → 40 × 40
        transforms.RandomCrop(32),  # random 32 × 32 crop
        transforms.RandomHorizontalFlip(),  # 50 % chance to flip left↔right
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.4465], std=[0.202, 0.1994, 0.2010]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.4465], std=[0.202, 0.1994, 0.2010]),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=train_transform, download=True
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=test_transform, download=True
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=True, num_workers=2
)

# set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
num_epochs = 50
learning_rate = 0.1
loss_function = nn.CrossEntropyLoss()
error = []


def train(model):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[81, 163],  # epoch numbers
        gamma=0.1,
    )

    for epoch in range(num_epochs):
        model.train()
        loss = torch.tensor([1])
        start = time.time()
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # perform a forward pass
            loss = loss_function(outputs, labels)  # calculate loss for the mini_batch

            optimizer.zero_grad()  # zero out the previously computed gradients
            loss.backward()  # compute the gradients
            optimizer.step()  # update the parameters of the NN

        scheduler.step()
        end = time.time()
        print(
            f"Epoch = {epoch + 1} / {num_epochs}, loss = {loss.item(): .4f}, time taken = {end - start: .4f} seconds"
        )

        # testing for this epoch
        model.eval()
        with torch.no_grad():
            correct = 0
            total = testset.data.shape[0]
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                _, predicted = torch.max(output, 1)
                correct += (predicted == labels).sum().item()
        error.append((1 - correct / total) * 100)
        print(f"Accuracy = {correct * 100 / total}")


def plot(n):
    iteration = range(len(error))
    plt.plot(iteration, error, color="gold", linewidth=2, label=f"ResNet-{6*n+2}")

    plt.ylabel("error(%)")
    plt.xlabel("epoch")
    plt.title("Test Error vs Epoch for ResNet Variants")
    plt.legend(loc="best")

    plt.axhline(y=20, color="grey", linestyle="--", linewidth=1)
    plt.axhline(y=15, color="grey", linestyle="--", linewidth=1)
    plt.axhline(y=10, color="grey", linestyle="--", linewidth=1)
    plt.axhline(y=5, color="grey", linestyle="--", linewidth=1)

    plt.yticks(range(0, 21, 5))
    plt.tight_layout()
    plt.savefig(f"./resnet{6*n+2}_error_curve.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # for n there will be 6n + 2 layers in the resnet
    n = 3  # resnet20
    model = ResNet(ResidualBlock, n).to(device)
    train(model)
    save_path = f"./data/resnet{6*n+2}_cifar10.pth"
    torch.save(model.state_dict(), save_path)
    plot(n)
