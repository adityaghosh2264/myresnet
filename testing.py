from myresnet import ResNet, ResidualBlock
import torch, torchvision, torchvision.transforms as transforms, torch.nn as nn
import time
import matplotlib.pyplot as plt

#load dataset and transform it for training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.4465],
                         std = [0.202, 0.1994, 0.2010])
])

trainset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform, download = True)
testset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = transform, download = True)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size = 128, shuffle = True, num_workers = 2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size = 128, shuffle = True, num_workers = 2
)

#set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
num_epochs = 50
learning_rate = 0.1
loss_function = nn.CrossEntropyLoss()
error = []

def train(model):
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.0001, momentum = 0.9)
    for epoch in range(num_epochs):
      model.train()
      loss = torch.tensor([1])
      start = time.time()
      for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images) #perform a forward pass
        loss = loss_function(outputs, labels) # calculate loss for the mini_batch

        optimizer.zero_grad() # zero out the previously computed gradients
        loss.backward() # compute the gradients
        optimizer.step() # update the parameters of the NN

      end = time.time()
      print(f'Epoch = {epoch +1} / {num_epochs}, loss = {loss.item(): .4f}, time taken = {end - start: .4f} seconds')

      #testing for this epoch
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
      error.append((1 - correct/total)*100)
      print(f'Accuracy = {correct*100 / total}')

def plot():
    iteration = range(len(error))
    plt.plot(iteration, error, color = 'gold', linewidth = 2, label = 'ResNet-20')

    plt.ylabel('error(%)')
    plt.xlabel('epoch')
    plt.title('Test Error vs Epoch for ResNet Variants')
    plt.legend(loc='best')

    plt.axhline(y=20, color='grey', linestyle='--', linewidth=1)
    plt.axhline(y=15, color='grey', linestyle='--', linewidth=1)
    plt.axhline(y=10, color='grey', linestyle='--', linewidth=1)
    plt.axhline(y=5, color='grey', linestyle='--', linewidth=1)

    plt.yticks(range(0, 21, 5))
    plt.tight_layout()
    plt.savefig('./resnet_error_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #for n there will be 6*n + 2 layers in the resnet
    n =  #resnet20
    model = ResNet(ResidualBlock, n).to(device)
    train(model)
    plot()