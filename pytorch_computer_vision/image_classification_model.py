
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from timeit import default_timer as timer


BATCH_SIZE = 32 # normally people choose powers of 2

#the object train_data will have data(tensor), targets(tensor) and classes(list) stored as attributes
train_data = datasets.FashionMNIST(root = "/home/ajamshid/sgoinfre/FashionMNIST",
                                   train=True,
                                   download=True,
                                   transform=ToTensor(),
                                   target_transform=None
)
test_data = datasets.FashionMNIST(root="/home/ajamshid/sgoinfre/FashionMNIST",
                                  train=False,
                                  download=True,
                                  transform=ToTensor())

# image, label = train_data[0]
# print(train_data.classes[label])
# print(f"Image shape: {image.shape}")
# plt.imshow(image.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
# plt.title(label)
# plt.show()

train_dataloader = DataLoader(train_data,
                              batch_size = BATCH_SIZE,
                              shuffle = True)

test_dataloader = DataLoader(test_data,
                             batch_size = BATCH_SIZE,
                             shuffle = True)
# print(len(train_dataloader), len(test_dataloader))



class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # keeps the first dim and reshapes the rest into a single dimention
            nn.Linear(in_features= input_shape, out_features= hidden_units),
            nn.Linear(in_features= hidden_units, out_features= output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)
    
model_0 = FashionMNISTModelV0(input_shape= 784,
                              hidden_units=10,
                              output_shape=len(train_data.classes))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params= model_0.parameters(), lr=0.1)

def print_train_time(start:float, end:float):
    """Prints difference between start and end time.
    
    Args:
        start (fload): start time of the computation (in timeit formate)
        end (float : end time of the computation)
        
    returns:
        float: time beween start and end in seconds
    """
    total_time = end - start
    print(f"Train time : {total_time:.3f} seconds")
    return total_time

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

torch.manual_seed(42)




print("Testing preformance of the model before training")
test_loss, test_acc = 0,0
with torch.inference_mode():
    for X, y in test_dataloader:
        test_pred = model_0(X)
        test_loss += loss_fn(test_pred, y)
        test_acc += accuracy_fn(y, test_pred.argmax(dim = 1))
    test_loss /= len(test_dataloader) # we go though this just to make it dynamic
    test_acc /= len(test_dataloader)
print(f"\nTest loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

print("training the model")
epochs = 3
train_time_start = timer()
for epoch in range(epochs):
    print(f"epoch: {epoch}\n----------")
    
    #training
    train_loss = 0 # this will be used to calculate avrage loss while training since we train the model in batches
    
    for batch, (X,y) in enumerate(train_dataloader): # batch would be the counter added by enumerate
        model_0.train()
        y_pred = model_0(X)
        #calculated loss per batch
        loss = loss_fn(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 400 == 0:
            print(F"looked at {batch *len(X)}/{len(train_dataloader.dataset)} samples")
    train_loss /= len(train_dataloader)
    
    test_loss, test_acc = 0,0
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            test_pred = model_0(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y, test_pred.argmax(dim = 1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

train_time_end = timer()
total_train_time_model_0 = print_train_time(start=train_time_start,
                                           end=train_time_end)
print(f"end result of the model after trainng for {epochs} epochs in {total_train_time_model_0:.3f} seconds :",
      f"model_name: {model_0.__class__.__name__}",
      f"model_loss: {test_loss}",
      f"model_acc: {test_acc}", sep = "\n")

