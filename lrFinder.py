from lr_finder import LRFinder
from models.OneDCNN import OneDCNN
from models.HypeTrain import Conv1DModelWithTransformers
from models.BidirectionalLSTM import Conv1DModelWithLSTM
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from Preprocess import H5Dataset
from torch import optim
import torch
import os
from pathlib import Path

# use cuda
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = Conv1DModelWithTransformers()
criterion = sigmoid_focal_loss
optimizer = optim.Adam(model.parameters(), lr=1e-7) #, weight_decay=1e-2)
pat_dir = '/mnt/wines/intra/original_data/inv/pat_26402'
h5_dir = Path(pat_dir + os.sep + 'downsampled_2' + os.sep + 'ieeg')
dataset = H5Dataset(h5_dir)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

lr_finder = LRFinder(model, optimizer, criterion, device="cuda:2")
lr_finder.range_test(train_loader, end_lr=10, num_iter=30000)
ax = lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state

# save teh ax object to a file
print(ax)
ax[0].get_figure().savefig('lr_finder.png')
