import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
import cnn

trainset = MNIST('.', train=True, download=True, transform=ToTensor())
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = MNIST(".", train=False, download=True, transform=ToTensor())
testloader = DataLoader(testset, batch_size=64, shuffle=False)


lr = 1e-4
num_epochs = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = cnn.MNISTConvNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

def train(optimizer=optimizer,
          model=model,
          num_epochs=num_epochs,
          loss_fn=loss_fn):
    print(f"Training on {device}")
    model.train()
    for epochs in range(num_epochs):
        running_loss = 0.0
        num_correct = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            _, idx = outputs.max(dim=1)
            num_correct += (idx == labels.to(device)).sum().item()
        print('Loss: {} Accuracy: {}'.format(running_loss/len(trainloader),
                num_correct/len(trainloader)))
    torch.save(model.state_dict(), "cnn_mnist_500.pt")
        
def main():
    train()

if __name__ == "__main__":
    main()