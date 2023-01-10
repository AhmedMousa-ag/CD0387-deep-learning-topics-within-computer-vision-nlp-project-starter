#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import argparse

from smdebug import modes
from smdebug.pytorch import get_hook

def test(model, test_loader,criterion,device,hook):

    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    hook.register_loss(criterion)
    for inputs, labels in test_loader:
        hook.set_mode(modes.EVAL)
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    
    pass

def train(model, train_loader, criterion, optimizer,device,epochs,hook):

    hook.register_loss(criterion)
    model.train()
    for e in range(epochs):
        hook.set_mode(modes.TRAIN)
        running_loss=0
        correct=0
        for data, target in train_loader:
            data=data.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            pred = model(data)             #No need to reshape data since CNNs take image inputs
            loss = criterion(pred, target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
            Accuracy {100*(correct/len(train_loader.dataset))}%")
    
    
def net():

    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 10))
    return model
    

def create_data_loaders(data, batch_size):

    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True
    )

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model=net()
    model=model.to(device)


    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, momentum=args.momentum)
    hook = get_hook(create_if_not_exists=True)


    data_lengh = len(args.data)
    train_len = int(data_lengh*0.8)
    test_len = int(data_lengh*0.2)
    train_data,test_data = torch.utils.data.random_split(args.data, [train_len, test_len])
    train_loader = create_data_loaders(train_data,args.batch_size) 
    model=train(model, train_loader, loss_criterion, optimizer,device,args.epochs,hook)

    test_loader = create_data_loaders(test_data,args.test_batch_size,hook)
    criterion = nn.CrossEntropyLoss()
    test(model, test_loader, criterion, device,)

    path = args.save_path
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--save_path", type=str, help="The path where trained model will be saved"
    )

       
    args=parser.parse_args()
    main(args) # Will call train and test
