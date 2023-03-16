# This is a sample Python script.
import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch import optim
from torch.utils.data import Dataset
from torchvision.ops import sigmoid_focal_loss
# import KFold from scikit learn
from sklearn.model_selection import KFold
# from models.OneDCNN import OneDCNN
from models.HypeTrain import Conv1DModelWithTransformers
# from models.BidirectionalLSTM import Conv1DModelWithLSTM
from Preprocess import H5Dataset
# import the python native path lib
from pathlib import Path
from glob import glob
import os
import json

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(DEVICE)
ALPHA = 0.5
GAMMA = 2

MLPROJECTRESULTS_DIR = Path('/mnt/wines/dschonholtz/MLProjectResults')

RESULTS_DIR = Path(MLPROJECTRESULTS_DIR / 'resultsCNNTransformer')


def dew_it(pat_dir):
    results = {}
    pat_key = pat_dir.split('/')[-1]
    results[pat_key] = {
        'epochs': [],
        'num_samples': 0,
        'non_results_files': [],
        'non_results': [],
        'pre_results_files': [],
        'pre_results': []
    }
    h5_dir = Path(pat_dir + os.sep + 'downsampled_2' + os.sep + 'ieeg')
    dataset = H5Dataset(h5_dir)
    results[pat_key]['num_samples'] = len(dataset)
    # set the cuda device
    # model = OneDCNN()
    # model = NaiveTransformerClassifier()
    # model = Conv1DModelWithLSTM()
    model = Conv1DModelWithTransformers()
    # Define your hyperparameters
    lr = 6.5e-06  # optimal learning rate is 0.0002 ish for cnn 6.51519e-06 for lstm and transformer
    # lr = lr / (len(dataset) / 543576)
    epochs = 15  # number of epochs 5 for cnn 10 for lstm
    batch_size = 32  # batch size

    # Define your optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = sigmoid_focal_loss

    # Define your learning rate scheduler that reduces lr when it plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=1, verbose=True)

    # Create an instance of the KFold class with 5 splits
    kf = KFold(n_splits=10, shuffle=True)
    # Loop over each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        # Print the current fold number
        print(f"Fold {fold + 1}")
        # Get the train and test subsets from the dataset using the indices
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)
        # Create dataloaders for the train and test subsets
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)
        print('Data loaders built')

        # Train and test the model for each epoch using the functions defined above
        for epoch in range(epochs):
            # Train the model on the train subset and get the loss and accuracy
            train_loss, train_acc = train(model, train_loader, optimizer, criterion)
            # Print the loss and accuracy for this epoch
            print(f"Epoch {epoch + 1}, Loss: {train_loss}, Accuracy: {train_acc}")
            results[pat_key]['epochs'].append((train_loss, train_acc))

            # Test the model on the test subset and get the loss and accuracy
            test_loss, test_acc = test(model, test_loader, criterion)

            # Print the loss and accuracy for this fold
            print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
            results[pat_key]['epochs'].append((test_loss, test_acc))

            # Update the learning rate scheduler based on the test loss
            scheduler.step(test_loss)
        # Lol only one fold
        break

    results[pat_key] = results_in_order(dataset, results[pat_key], model)
    print('saving results')
    with open(f'{RESULTS_DIR}/{pat_key}.json', 'w') as f:
        json.dump(results, f)
    # save the model
    torch.save(model.state_dict(), MLPROJECTRESULTS_DIR / f'models/cnn_tranformer_{pat_key}.pt')


def results_in_order(dataset, results, model):
    # go through all of the sorted data and save the model results to a file.
    with torch.no_grad():
        all_results = []
        pre_subset = torch.utils.data.Subset(dataset, range(dataset.non_idx))
        non_subset = torch.utils.data.Subset(dataset, range(dataset.non_idx, len(dataset)))
        pre_loader = torch.utils.data.DataLoader(pre_subset, batch_size=128)
        non_loader = torch.utils.data.DataLoader(non_subset, batch_size=128)
        filenames = [x.split(os.sep)[-1] for x in dataset.h5_files]
        channels_per_file = dataset.file_channels
        results['files'] = filenames
        results['channels_per_file'] = channels_per_file
        loaders = [pre_loader, non_loader]
        loader_idx = 0
        model.to(DEVICE)
        for loader in loaders:
            for i, (data, labels) in enumerate(loader):
                data = data.to(DEVICE).float()
                labels = labels.to(DEVICE).float()
                # Forward pass
                outputs = model(data).cpu().detach().numpy()
                result = [0 if output[0] > output[1] else 1 for output in outputs]
                if loader_idx == 0:
                    results['pre_results_files'] += result
                else:
                    results['non_results_files'] += result
            loader_idx += 1
            print('Finished one loader')
        return results


def main():
    print('starting')
    result_files = glob(f'{RESULTS_DIR}/*.json')
    pat_dirs = glob('/mnt/wines/intra/original_data/inv/pat_*')
    pat_dirs += glob('/mnt/wines/intra/original_data/inv2/pat_*')
    # hard_pats = [
    #     'pat_26402',
    #     'pat_54802',
    #     'pat_25302',
    #     'pat_37502',
    #     'pat_112502',
    #     'pat_63502'
    #     'pat_109602'
    # ]
    for pat_dir in pat_dirs:
        pat_key = pat_dir.split(os.sep)[-1]
        # if f'{RESULTS_DIR}/{pat_key}.json' in result_files:
        #     print(f'{pat_key} already done')
        #     continue
        # if pat_key not in hard_pats:
        #     print(f'{pat_key} is ez')
        #     continue
        dew_it(pat_dir)
    # dew_it('/mnt/wines/intra/original_data/inv/pat_26402') # 543,576 samples
    # dew_it('/mnt/wines/intra/original_data/inv/pat_114602') # 4,080,858 samples


# Define a function to train the model on a given dataloader
def train(model, dataloader, optimizer, criterion):
    # Set the model to training mode
    model = model.to(DEVICE)
    model.train()
    # Initialize the running loss and accuracy
    train_loss = 0.0
    train_acc = 0.0
    # Loop over the batches of the dataloader
    for data, labels in dataloader:
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(data)
        # Compute the loss
        loss = criterion(outputs.squeeze(), labels, alpha=ALPHA, gamma=GAMMA)
        # Backward pass and optimize
        loss.backward(torch.ones_like(outputs))
        optimizer.step()
        # Update the running loss and accuracy
        # iterate over the batch size
        for i in range(len(loss)):
            train_loss += loss[i][0].item() + loss[i][1].item()
        train_acc += ((outputs > 0.5).squeeze() == labels).float().mean().item()
    # Return the average loss and accuracy
    return train_loss / len(dataloader), train_acc / len(dataloader)


# Define a function to test the model on a given dataloader
def test(model, dataloader, criterion):
    # Set the model to evaluation mode
    model = model.to(DEVICE)
    model.eval()
    # Initialize the test loss and accuracy
    test_loss = 0.0
    test_acc = 0.0
    # Loop over the batches of the dataloader
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            # Forward pass
            outputs = model(data)
            # Compute the loss
            loss = criterion(outputs.squeeze(), labels, alpha=ALPHA, gamma=GAMMA)
            # Update the test loss and accuracy
            for i in range(len(loss)):
                test_loss += loss[i][0].item() + loss[i][1].item()
            test_acc += ((outputs > 0.5).squeeze() == labels).float().mean().item()
    # Return the average loss and accuracy
    return test_loss / len(dataloader), test_acc / len(dataloader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
