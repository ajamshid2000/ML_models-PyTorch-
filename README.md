to copile .py files you can use pyhone/python3 filename.py

in this repo we will have multiple models such as strait_line_predictor, binary_classification, multi_class_classification, computer_vision etc.

1. strait_line_predictor: the model will predict the y position of the dots making a strait line by input of its x coordinate, in this project we go throught:
*creation of a range of x tensors, calculating the y coordinate using wx+b and spliting the data to training(80%) and testing(20%) data.
*creating a plot function for visualization of our data.
*creating a custom model and visualizing the initial model predictions.
*choosing the right Loss and Optemizer funcitons.
*training the model for 2500 epochs with reducing learning rate each 200 epochs for optemization of our model and printing the info on training each 10 epochs.
*finally dispaying the loss curves and final prediction after 2500 epochs of learning

