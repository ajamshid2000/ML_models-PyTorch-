import torch
from torch import nn
import matplotlib.pyplot as plt

print(torch.__version__)

# below we will create data set for linear regression model

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02 

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8*len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# we make a plot function to visualize the data on a graph
def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_label =y_test,
                     predictions = None):
    """
    plots training data, test data, and compares predeictions
    """
    plt.figure(figsize=(5,4))
    
    # plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label = "Training data")
    
    # plot test data in green 
    plt.scatter(test_data, test_label, s= 4, c = "g", label = "Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c= "r", s = 4, label = "predictions")
        
    plt.legend(prop = {"size": 14})
    plt.show()

# now we will make a custom model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wights = nn.Parameter(torch.randn(1, dtype=float),
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,dtype=float,
                                            requires_grad=True))
    def forward(self, x:torch.tensor) -> torch.tensor:
        return self.wights*x + self.bias

# we can set manual seed to have the same random numbers
# torch.manual_seed(42)
model_0 = LinearRegressionModel()

# Make predictions with model in iference mode to turn of unnessesary operations since we will not be training the model
with torch.inference_mode():
    y_preds = model_0(X_test)

# lets see how our predictions have turned out
plot_predictions(predictions=y_preds)

# Mean absolute error (MAE) for regression problems (torch.nn.L1Loss()). Binary cross entropy for binary classification problems (torch.nn.BCELoss()).
# Stochastic gradient descent (torch.optim.SGD()). Adam optimizer (torch.optim.Adam()).
# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 2500

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []
learning_rate = 0.01

for epoch in range(epochs):
    ### Training
    if(epoch%200 == 0):
        # i would like to decrese the learning rate each 100 epocs to learn better
        learning_rate /= 1.2
        optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=learning_rate)
        print(learning_rate)

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()


# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

plot_predictions(predictions=test_pred)





