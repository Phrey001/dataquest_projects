"""
Introduction:

In this Guided Project, we'll:
    - explore why image classification is a hard task.
    - observe the limitations of traditional machine learning models
    for image classification.
    - train, test, and improve a few different deep neural networks
    for image classification.
    
Image classification is a hard task:
    - Each image in a training set is high dimensional.
    Each pixel in an image is a feature and a separate column.
    This means that a 128 x 128 image has 16384 features.
    - images are often downsampled to lower resolutions and transformed
    to grayscale (no color). This is a limitation of computation power. 
    -  features in an image don't have an obvious linear or nonlinear
    relationship that can be learned with a model like linear
    or logistic regression. In grayscale, each pixel is just represented
    as a brightness value ranging from 0 to 256.
"""

"""
Working With Image Data:
    Scikit-learn contains a number of datasets pre-loaded with the library,
    within the namespace of sklearn.datasets.
    The load_digits() function returns a copy of the hand-written digits
    dataset from UCI.
"""
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

digits_data = load_digits()  ## Load dataset from UCI
print(digits_data.keys())  ## preview keys of dataset

labels = pd.Series(digits_data['target'])  ## assign "target" attribute to variable
data = pd.DataFrame(digits_data['data'])  ## assign data to dataframe
print(data.head(1))  ## preview data

"""
## To reshape the image, we need to convert a training example
to a numpy array (excluding the label column) and pass the result into
that into the numpy.reshape() function.

## reshape and plot first image from 1-D data on greyscale:    
    - assign first image data to variable.
    - convert variable to np array format before reshape.
    - reshape np array from 1-D to multi-D dimensions (eg. 8x8) for pictures.
    - After data is in the right shape, visualize it using pyplot.imshow().

## reshape and plot multiple images from 1-D data on greyscale:
    - repeat process to plot other image sample data on same plot
    (2 rows, 4 columns).
    - 1st, 100th, 200th, 300th, 1000th, 1100th, 1200th, 1300th row data
"""
## reshape and plot first image from 1-D data on greyscale
first_image = data.iloc[0]  
np_image = first_image.values  
np_image = np_image.reshape(8,8)  
plt.imshow(np_image, cmap='gray_r')  
## reshape and plot multiple images from 1-D data on greyscale
f, axarr = plt.subplots(2, 4)  # 2 rows, 4 columns on figure, axes array (f, axarr)
axarr[0, 0].imshow(data.iloc[0].values.reshape(8,8), cmap='gray_r')
axarr[0, 1].imshow(data.iloc[99].values.reshape(8,8), cmap='gray_r')
axarr[0, 2].imshow(data.iloc[199].values.reshape(8,8), cmap='gray_r')
axarr[0, 3].imshow(data.iloc[299].values.reshape(8,8), cmap='gray_r')
axarr[1, 0].imshow(data.iloc[999].values.reshape(8,8), cmap='gray_r')
axarr[1, 1].imshow(data.iloc[1099].values.reshape(8,8), cmap='gray_r')
axarr[1, 2].imshow(data.iloc[1199].values.reshape(8,8), cmap='gray_r')
axarr[1, 3].imshow(data.iloc[1299].values.reshape(8,8), cmap='gray_r')
"""
K-Nearest Neighbors Model:
    The k-nearest neighbors algorithm compares every unseen observation in
    the test set to all (or many, as some implementations constrain the
    search space) training observations to look for similar (or the "nearest")
    observations. Then, the algorithm finds the label with the most nearby
    observations and assigns that as the prediction for the unseen observation.
    
    Use the KNeighborsClassifier package to train and test k-nearest neighbors
    models.
    
There are a few downsides to using k-nearest neighbors:
    - high memory usage (for each new unseen observation,
                         many comparisons need to be made to seen observations)
    - no model representation to debug and explore

# 50% Train / test validation
# Use defined functions to modularise the process
so that easy to repeat/debug process on different iterations when needed:
    - train() that uses KNeighborsClassifer for
    training k-nearest neighbors models.
    - test() tests the model
    - cross_validate() that performs N-fold cross validation
    using train() and test().
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

## for later usage in cross-validation function; train the model with train data
def train_knn(nneighbors, train_features, train_labels):
    knn = KNeighborsClassifier(n_neighbors = nneighbors)
    knn.fit(train_features, train_labels)
    return knn

## for later usage in cross-validation function; test the model with test data
def test(model, test_features, test_labels):
    predictions = model.predict(test_features)
    train_test_df = pd.DataFrame()
    train_test_df['correct_label'] = test_labels
    train_test_df['predicted_label'] = predictions
    overall_accuracy = sum(train_test_df["predicted_label"] ==
                           train_test_df["correct_label"])/len(train_test_df)    
    return overall_accuracy

def cross_validate_knn(k, n_splits=4):
    """
    Parameters
    ----------
    k : integer
        Number of nearest neighbours to apply to KNN clustering model

    Returns
    -------
    fold_accuracies : float
        Measures accuracy of model using defined K-Fold validation, where K
        is number of n_splits. This will output list of accuracy depending on
        number of n_splits used. Hence, overall accuracies may be 
        computed by taking average mean later.
    """
    fold_accuracies = []
    kf = KFold(n_splits = n_splits, random_state=2, shuffle=True)  ## Instantiate the classifer to perform train-test-split using KFold library, input parameters for KFold
    for train_index, test_index in kf.split(data):  ## Applying kf.split() function on dataset to get split data's train and test index on each loop.
        train_features, test_features = data.loc[train_index], data.loc[test_index]  # define train_features, test_features from data
        train_labels, test_labels = labels.loc[train_index], labels.loc[test_index]  # define train_features, test_features from data target labels
        model = train_knn(k, train_features, train_labels)  ## Apply train_knn function here
        overall_accuracy = test(model, test_features, test_labels)  ## Apply test function here
        fold_accuracies.append(overall_accuracy)
    return fold_accuracies
        
knn_one_accuracies = cross_validate_knn(1)  # 1 means 1 nearest neighbour in KNN algo, and output list of KFold accuracy from n_splits.
print(np.mean(knn_one_accuracies))  # output test accuracy by taking average mean from list of KFold accuracy.

"""
## Loop and compute test accuracy using range of inputs 1 to 9 nearest
neighbours in KNN algo, including taking average mean of each KFold with
n_split iteration.
## Plot the accuracy output results.
"""
k_values = list(range(1,10))
k_overall_accuracies = []

for k in k_values:
    k_accuracies = cross_validate_knn(k)
    k_mean_accuracy = np.mean(k_accuracies)
    k_overall_accuracies.append(k_mean_accuracy)
    
plt.figure(figsize=(8,4))
plt.title("Mean Accuracy vs. k")
plt.plot(k_values, k_overall_accuracies)
plt.show()

"""
Neural Network with One Hidden Layer:
    Use the MLPClassifier package from scikit-learn.
"""
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

# 50% Train / test validation
def train_nn(neuron_arch, train_features, train_labels):
    """
    Parameters
    ----------
    neuron_arch : Integer
        Number of hidden layers in Neural Network Model (
            Each hidden layer can have multiple neurons).
    train_features : Float
        Input split data from train dataset later.
    train_labels : Float
        Input split data target labels from train dataset later.

    Returns
    -------
    mlp : TYPE
        Return the trained model after fit.
    """
    mlp = MLPClassifier(hidden_layer_sizes=neuron_arch)   ## Instantiate MLPClassifier Class with input parameters
    mlp.fit(train_features, train_labels)
    return mlp

def test(model, test_features, test_labels):
    """
    Predict the model using test features input from test dataset later.
    
    Returns overall_accuracy after comparing
    target labels between prediction and test dataset.
    """
    predictions = model.predict(test_features)
    train_test_df = pd.DataFrame()
    train_test_df['correct_label'] = test_labels
    train_test_df['predicted_label'] = predictions
    overall_accuracy = sum(train_test_df["predicted_label"] == train_test_df["correct_label"])/len(train_test_df)    
    return overall_accuracy

def cross_validate_neural(neuron_arch, n_splits=4):
    fold_accuracies = []
    kf = KFold(n_splits = n_splits, random_state=2, shuffle=True)  ## Instantiate KFold Class with input parameters for cross-validation
    for train_index, test_index in kf.split(data):  ## On each loop, get index of each split data on train and test dataset
        train_features, test_features = data.loc[train_index], data.loc[test_index]
        train_labels, test_labels = labels.loc[train_index], labels.loc[test_index]
       
        model = train_nn(neuron_arch, train_features, train_labels)  ## Apply train_nn() function here on train split data
        overall_accuracy = test(model, test_features, test_labels)  ## Apply test() function here on test split data
        fold_accuracies.append(overall_accuracy)
    return fold_accuracies


"""
## Define number of neurons in each hidden layer for the Neural Network Model as input.
"""
nn_one_neurons = [
    (8,),
    (16,),
    (32,),
    (64,),
    (128,),
    (256,)
]
nn_one_accuracies = []

for n in nn_one_neurons:
    nn_accuracies = cross_validate_neural(n, n_splits=4)  ## result in list of accuracy from KFold cross-validation
    nn_mean_accuracy = np.mean(nn_accuracies)  ## Take mean average
    nn_one_accuracies.append(nn_mean_accuracy)

plt.figure(figsize=(8,4))
plt.title("Mean Accuracy vs. Neurons In Single Hidden Layer")

x = [i[0] for i in nn_one_neurons]
plt.plot(x, nn_one_accuracies)
plt.show()

"""
## Summary Write-up:
    It looks like adding more neurons to the single hidden layer improved
    simple accuracy to approximately 97%. Simple accuracy computes the number
    of correct classifications the model made, but doesn't tell us anything
    about false or true positives or false or true negatives.
    
    Given that k-nearest neighbors achieved approximately 98% accuracy,
    there doesn't seem to be any advantages to using a single hidden
    layer neural network for this problem.
"""

"""
Neural Network with Two Hidden Layers:
    Iterate Neural Network Model but with 2 hidden layers.
"""
nn_two_neurons = [
    (64,64),  ## eg. 64 neurons in 1st hidden later, 64 in 2nd hidden later
    (128, 128),
    (256, 256)
]
nn_two_accuracies = []

for n in nn_two_neurons:
    nn_accuracies = cross_validate_neural(n, n_splits=4)  ## result in list of accuracy from KFold cross-validation
    nn_mean_accuracy = np.mean(nn_accuracies)  ## take mean average
    nn_two_accuracies.append(nn_mean_accuracy)

plt.figure(figsize=(8,4))
plt.title("Mean Accuracy vs. Neurons In Two Hidden Layers")

x = [i[0] for i in nn_two_neurons]
plt.plot(x, nn_two_accuracies)
plt.show()
print(nn_two_accuracies)

"""
## Summary Write-up:
    Using two hidden layers improved our simple accuracy to 98%.
    While, traditionally, we might worry about overfitting,
    using four-fold cross validation also gives us a bit more assurance
    that the model is generalizing well on test dataset to achieve the
    extra 1% in simple accuracy over the single hidden layer networks
    we tried earlier.
    
    Generalization is the ability of a neural network to correctly recognize
    patterns of input data that were not present in the training data.
    This is a critical property of neural networks,
    as it allows them to be used for tasks such as classification,
    prediction, and optimization.
"""

"""
Neural Network with Three Hidden Layers:
    Iterate Neural Network Model but with 3 hidden layers.
"""
nn_three_neurons = [
    (10, 10, 10),  ## 10/10/10 in 1st,2nd,3rd hidden layer
    (64, 64, 64),  ## 64/64/64 in 1st,2nd,3rd hidden layer
    (128, 128, 128)  ## 128/128/128 in 1st,2nd,3rd hidden layer
]

nn_three_accuracies = []

for n in nn_three_neurons:
    nn_accuracies = cross_validate_neural(n, n_splits=6)
    nn_mean_accuracy = np.mean(nn_accuracies)
    nn_three_accuracies.append(nn_mean_accuracy)

plt.figure(figsize=(8,4))
plt.title("Mean Accuracy vs. Neurons In Three Hidden Layers")

x = [i[0] for i in nn_three_neurons]
plt.plot(x, nn_three_accuracies)
plt.show()

print(nn_three_accuracies)

"""
## Summary Write-up:
    Using three hidden layers returned a simple accuracy of nearly 98%,
    even with six-fold cross validation.
"""