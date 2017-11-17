import torch
import torchvision.datasets as dsets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
import shutil
import os.path
import time
import numpy as np

# Hyperparameter
batch_size = 128
input_size = 784  # 28 * 28
hidden_size = 500
num_classes = 10
learning_rate = 1e-3
num_epochs = 12
print_every = 100
best_accuracy = torch.FloatTensor([0])
start_epoch = 0

# Path to saved model weights(as hdf5)
resume_weights = "/model/checkpoint.pth.tar"

# CUDA?
cuda = torch.cuda.is_available()

# Seed for reproducibility
torch.manual_seed(1)
if cuda:
	torch.cuda.manual_seed(1)


def train(model, optimizer, train_loader, test_loader, loss_fn):
	"""Perform a full training over dataset"""
	average_time = 0
	# Model train mode
	model.train()
	for i, (images, labels) in enumerate(train_loader):
		# measure data loading time
		batch_time = time.time()
		images = Variable(images)
		labels = Variable(labels)

		if cuda:
			images, labels = images.cuda(), labels.cuda()

		# Forward + Backward + Optimize
		optimizer.zero_grad()
		outputs = model(images)
		loss = loss_fn(outputs, labels)

		# Load loss on CPU
		if cuda:
			loss.cpu()

		loss.backward()
		optimizer.step()

		# Measure elapsed time
		batch_time = time.time() - batch_time
		# Accumulate over batch
		average_time += batch_time

		# ### Keep track of metric every batch
		# Accuracy Metric
		prediction = outputs.data.max(1)[1]   # first column has actual prob.
		accuracy = prediction.eq(labels.data).sum() / batch_size * 100

		# Log
		if (i + 1) % print_every == 0:
			print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Accuracy: %.4f, Batch time: %f'
				% (epoch + 1,
					num_epochs,
					i + 1,
					len(train_dataset) // batch_size,
					loss.data[0],
					accuracy,
					average_time/print_every))  # Average


def eval(model, optimizer, test_loader):
	"""Eval over test set"""
	model.eval()
	correct = 0
	# Get Batch
	for data, target in test_loader:
		data, target = Variable(data, volatile=True), Variable(target)
		if cuda:
			data, target = data.cuda(), target.cuda()
		# Evaluate
		output = model(data)
		# Load output on CPU
		if cuda:
			output.cpu()
		# Compute Accuracy
		prediction = output.data.max(1)[1]
		correct += prediction.eq(target.data).sum()
	return correct


def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
	"""Save checkpoint if a new best is achieved"""
	if is_best:
		print ("=> Saving a new best")
		torch.save(state, filename)  # save checkpoint
	else:
		print ("=> Validation Accuracy did not improve")


# MNIST Dataset (Images and Labels)
# If you have not mounted the dataset, you can download it
# just adding download=True as parameter
train_dataset = dsets.MNIST(root='/input',
							train=True,
							download=True,
							transform=transforms.ToTensor())
x_train_mnist, y_train_mnist = train_dataset.train_data.type(torch.FloatTensor), \
							train_dataset.train_labels
test_dataset = dsets.MNIST(root='/input',
							train=False,
							download=True,
							transform=transforms.ToTensor())
x_test_mnist, y_test_mnist = test_dataset.test_data.type(torch.FloatTensor), \
							test_dataset.test_labels

# Dataset info
print('Training Data Size: ', x_train_mnist.size(), '-', y_train_mnist.size())
print('Testing Data Size: ', x_test_mnist.size(), '-', y_test_mnist.size())

# Training Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											batch_size=batch_size,
											shuffle=True)
# Testing Dataset Loader (Input Pipline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
											batch_size=batch_size,
											shuffle=False)

# #### Model ####
# Convolutional Neural Network Model
class CNN(nn.Module):
	"""Conv[ReLU] -> Conv[ReLU] -> MaxPool -> Dropout(0.25)-
	-> Flatten -> FC()[ReLU] -> Dropout(0.5) -> FC()[Softmax]
	"""
	def __init__(self, num_classes):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
		self.drop1 = nn.Dropout2d(p=0.25)
		self.fc1 = nn.Linear(9216, 128)
		self.drop2 = nn.Dropout2d(p=0.5)
		self.fc2 = nn.Linear(128, num_classes)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = self.drop1(x)
		x = x.view(-1, 9216)
		x = F.relu(self.fc1(x))
		x = self.drop2(x)
		x = self.fc2(x)
		return F.log_softmax(x)

model = CNN(num_classes)
print(model)

# If you are running a GPU instance, load the model on GPU
if cuda:
	model.cuda()

# #### Loss and Optimizer ####
# Softmax is internally computed.
loss_fn = nn.CrossEntropyLoss()
# If you are running a GPU instance, compute the loss on GPU
if cuda:
	loss_fn.cuda()

# Set parameters to be updated.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# If exists a best model, load its weights!
if os.path.isfile(resume_weights):
	print("=> loading checkpoint '{}' ...".format(resume_weights))
	if cuda:
		checkpoint = torch.load(resume_weights)
	else:
		# Load GPU model on CPU
		checkpoint = torch.load(resume_weights,
								map_location=lambda storage,
								loc: storage)
	start_epoch = checkpoint['epoch']
	best_accuracy = checkpoint['best_accuracy']
	model.load_state_dict(checkpoint['state_dict'])
	print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights,
		checkpoint['epoch']))


# Training the Model
for epoch in range(num_epochs):
	train(model, optimizer, train_loader, test_loader, loss_fn)
	acc = eval(model, optimizer, test_loader)
	acc = 100. * acc / len(test_loader.dataset)
	print('=> Test set: Accuracy: {:.2f}%'.format(acc))
	acc = torch.FloatTensor([acc])
	# Get bool not ByteTensor
	is_best = bool(acc.numpy() > best_accuracy.numpy())
	# Get greater Tensor to keep track best acc
	best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))
	# Save checkpoint if is a new best
	save_checkpoint({
		'epoch': start_epoch + epoch + 1,
		'state_dict': model.state_dict(),
		'best_accuracy': best_accuracy
	}, is_best)
