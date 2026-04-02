import torch
import numpy as np
import cnn_train
import cnn

def load(filename):
    print("Loading file "+filename)
    model = cnn.MNISTConvNet()
    model.load_state_dict(torch.load(filename, weights_only=True))
    return model.eval()

def test(model=cnn_train.model, 
    loss_fn = cnn_train.loss_fn):
    print(f"testing on {cnn_train.device}")
    conf_matrix = np.zeros((10,10))
    model.eval()
    accuracy = 0.0
    computed_loss = 0.0
    with torch.no_grad():
        for data, target in cnn_train.testloader:
            out = model(data)
            _, preds = out.max(dim=1)
            for t, p in zip(target, preds):
                conf_matrix[t.item(), p.item()] += 1
            # Get loss and accuracy
            computed_loss += loss_fn(out, target)
            accuracy += torch.sum(preds==target)
            
        print("Test loss: {}, test accuracy: {}".format(
            computed_loss.item()/(len(cnn_train.testloader)*64), accuracy*100.0/(len(cnn_train.testloader)*64)))
        return conf_matrix

def display_conf_matrix(conf_matrix):
    print("CNN 40 Epochs MNIST Confusion Matrix")
    print("    ", end="")
    for i in range(10):
        print(i, end="     ")
    print()
    for i in range(10):
        print(i, end=" ")
        for j in range(10):
            print(f"{int(conf_matrix[i][j]):4d}", end="  ")
        print("\n")

def main():
    cmtx = test(model=load("cnn_mnist.pt"))
    display_conf_matrix(cmtx)

if __name__ == "__main__":
    main()