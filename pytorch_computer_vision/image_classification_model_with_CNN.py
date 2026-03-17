
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm import tqdm

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

train_dataloader = DataLoader(train_data,
                              batch_size = BATCH_SIZE,
                              shuffle = True)

test_dataloader = DataLoader(test_data,
                             batch_size = BATCH_SIZE,
                             shuffle = True)


class  FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, output_shape:int, hidden_units:int):
        super().__init__()
        self.block_0 = nn.Sequential(nn.Conv2d(input_shape, #4d shape (N, C, H, W) batch size, color channels, height, width
                                hidden_units,
                                kernel_size=3,
                                stride= 1,
                                padding= 1),
                      nn.ReLU(),
                      nn.Conv2d(hidden_units,
                                hidden_units,
                                kernel_size= 3,
                                stride= 1,
                                padding= 1),
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size= 3,
                                   stride= 2)
                      )
        self.block_1 = nn.Sequential(nn.Conv2d(hidden_units,
                                               out_channels=hidden_units,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=hidden_units,
                                               out_channels=hidden_units,
                                               kernel_size= 3,
                                               stride=1,
                                               padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3,
                                                  stride=2))
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features= hidden_units*6*6, out_features=output_shape))
    def forward(self, x: torch.tensor):
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.classifier(x)
        return x
        
torch.manual_seed(42)
model_0 = FashionMNISTModelV1(input_shape=1, 
    hidden_units=10, 
    output_shape=len(train_data.classes))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                             lr=0.1)


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy of a model.
    
    Args:
        y_true (int): true value
        y_pred (int) : predicted value
        
    returns:
        int: percentage of accuracy
    """
    accurate = torch.eq(y_pred, y_true).sum().item()
    return (accurate / len(y_pred) * 100)

def print_train_time(start, end):
    """Prints difference between start and end time.
    
    Args:
        start (fload): start time of the computation (in timeit formate)
        end (float : end time of the computation)
        
    returns:
        float: time beween start and end in seconds
    """
    total_time =  end - start
    print(f"Train time : {total_time:.3f} seconds")
    return total_time

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn, ):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

def train_step(data_loader:DataLoader, model:nn.Module, loss_fn:nn.Module, optimizer:torch.optim.Optimizer, accuracy_fn):
    """Trains the given model and prints out the train loss and accuracy.
    
    Args:
        data_loader (DataLoader) : batch generator of images
        model (nn.Module) : the ML model to train.
        loss_fn (nn.Module) : loss funtion for calculating loss of the model
        optimizer(torch.optim.Optimizer) : optemizer function to optimize the model
        accuracy_fn : accuracy function
    """
    train_loss, train_acc = 0, 0
    model.train()
    for X, y in data_loader:
        y_pred = model(X)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader:DataLoader, model:nn.Module, loss_fn:nn.Module, accuracy_fn):
    """Tests the given model and prints out the test loss and accuracy.
    
    Args:
        data_loader (DataLoader) : batch generator of images
        model (nn.Module) : the ML model to train.
        loss_fn (nn.Module) : loss funtion for calculating loss of the model
        accuracy_fn : accuracy function
    """
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for X, y in data_loader:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss
            test_acc += accuracy_fn(y_true = y,
                                    y_pred = y_pred.argmax(dim=1))
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")
        

torch.manual_seed(42)
train_time_start_model_0 = timer()

epochs = 3
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_0, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )
    test_step(data_loader=test_dataloader,
        model=model_0,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )

train_time_end_model_0 = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_model_0,
                                           end=train_time_end_model_0)


# Get model_2 results 
model_2_results = eval_model(
    model=model_0,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
print(model_2_results)

