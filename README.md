to copile .py files you can use python/python3 filename.py

in this repo we will have multiple models such as pytorch_linear_regression, binary_classification, multi_class_classification, computer_vision etc.

1. pytorch_linear_regression: the model will predict the y position of the dots making a straight line by input of its x coordinate, in this project we go throught:
  *creation of a range of x tensors, calculating the y coordinate using wx+b and spliting the data to training(80%) and testing(20%) data.
  *creating a plot function for visualization of our data.
  *creating a custom model and visualizing the initial model predictions.
  *choosing the right Loss and Optemizer funcitons.
  *training the model for 2500 epochs with reducing learning rate each 200 epochs for optemization of our model and printing the info on training each 10 epochs.
  *finally dispaying the loss curves and final prediction after 2500 epochs of learning

2. binary_classificaion_model: the model will classify the scatterd dots on two cirlces made by make_circles() from scikit_learn in this project we go through the step bellow:
   *creation of two circles using make_circles().
   *vistualization of the two cirlcles created.
   *splitting the data into train and test data using train_test_split() from scikit_learn
   *creating a model that can learn from the data.
   *choosing the right loss and optim functions.
   *creating an and accuracy funtion to calculate accuracy of predictions.
   *training the model for 1000 epochs.
   *building a visualization funtions to show the boundry of decision of the model.
   *finally dispalying the visualization of the data and boundry of decision for the test and train data.

3. multiclass_classification: the model will classify the scattered dots on patches made by make_blobs from scikit_learn, in this project we will go though:
   *creation of four patches of dots using make_blobs().
   *vistualization of the two cirlcles created.
   *splitting the data into train and test data using train_test_split() from scikit_learn
   *creating a model that can learn from the data.
   *choosing the right loss and optim functions.
   *creating an and accuracy funtion to calculate accuracy of predictions.
   *training the model for 1000 epochs.
   *building a visualization funtions to show the boundry of decision of the model.
   *finally dispalying the visualization of the data and boundry of decision for the test and train data.
