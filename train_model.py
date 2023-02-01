import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import argparse
import json
import logging
import os
import sys
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
import argparse

def test(model, test_loader,criterion,hook):
    if hook:
            hook.set_mode(modes.EVAL)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer,test_loader,args,hook):
    epoch_times = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        if hook:
            hook.set_mode(modes.TRAIN)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader,criterion,hook)
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
    p50 = np.percentile(epoch_times, 50)
    return p50
    
def net():

    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 111)) #TODO change it to the number of feature
    return model

def create_data_loaders(dir_path, batch_size):

    logger.info("Get train data loader")
    dataset = datasets.ImageFolder(dir_path, transform=transforms.Compose(
            [transforms.ToTensor(),transforms.Resize([240,240], 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=net()
    #model=model.to(device)

    '''
    TODO: Create your loss and optimizer
    '''
    hook = get_hook(create_if_not_exists=True)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if hook:
        hook.register_loss(loss_criterion)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_dir = os.path.join(args.data_dir,"train")
    train_loader= create_data_loaders(train_dir,args.batch_size,args)
    test_dir = os.path.join(args.data_dir,"test")
    test_loader= create_data_loaders(test_dir,args.batch_size)
    
    median_time=train(model, train_loader, loss_criterion, optimizer,test_loader,args,hook)
    print("Median training time per Epoch=%.1f sec" % median_time)

    '''
    TODO: Save the trained model
    '''
    save_model_path = os.path.join(args.model_dir,"best_model_dogs_breads.pth")

    torch.save(model.state_dict(), save_model_path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
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


    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--gpu", type=str2bool, default=True)


    args=parser.parse_args()
    
    main(args)

















