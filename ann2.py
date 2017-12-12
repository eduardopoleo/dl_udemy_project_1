# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#################### Pre Processing ################################
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical variables (from string to numbers)
# Encoding country
# Models do not understand strings but numbers so we need to transform this 
# categorical information into numerical labels.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Encoding Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# We have another problem now is that the model may think that 2 > 1 > 0 but
# in reality these are numerical labels not sequential values.
# An example of categorical data where this would actually makes sense 
# is t-shirt sizing.
# To prevent this we break stuff down into 3 columns all of them with 1 and 0
# we do that with the OneHotEncoder

# this will break down the countries into 3 dummy columns with corresponding 1,0
onehotencoder = OneHotEncoder(categorical_features = [1])
# not necessary for the gender cuz it's already binary
X = onehotencoder.fit_transform(X).toarray()

# Finally we need to remove one of the columns to prevent the 'Dummy variable trap'
# So happens that for France, Germany, Italy. We can determine if the occurrance
# happens in Italy if by knowing the France and Germany values. We do not want
# correlated columns in our dataset. So we need to eliminate one of them
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# what about validation set?


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Finds the parameters used to scale on the training set
X_train = sc.fit_transform(X_train)
# Uses the parameters derived in the training set to scale the test set. For some
# reason you do not want to re scale the test set.
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here


#################### NN fitting ###################################
# Part 2 - Now lets make the ANN!
# Import the KEras library and packages
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout  # used to reduce overfitting to the train model.

# Initializing the ANN
# This is the actual NN object to which we are adding the layers to
classifier = Sequential()

# The fastest activiation function is the RELU but the sigmoid function is good
# to express probability which is what we want. We'll use the sigmoid function
# for the last layer.


# Stochatic Gradient Descent. Modifies the weight after each (or some) of the
# rows are run. Regular Gradient Descent modifies the weight at the end.

# Hidden layer will have average of nodes between the input layer with the output
# layer, as a simplified architecture.

# output_dim Hidden layer dimentions
# init, Initialization values distribution
# activation, activation function

# This add the first layer and the first hidden layer
# input_dim, required for the first layer.
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))
# init only initializes the weights for the layer you're using
# output_dim is # of nodes
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# I do not understand why only one node. Shouldn't I have as many as labels there are?
# this is because the independent variables are 1 hot encoded that's why we used only 1
# IMPPPP: If I have several labels I need to use other activation function: softmax 
# which is the sigmoid function applied to all labels.
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# optimizer: algorightm that is used to find the weights. We'll use 'adam' which is a type of stochastic GD
# loss: in this case since we are using sigmoid. We'll use the logaritmic loss function.
# if 'categorical_crossentropy' if we have more than 1 label.
# metrics: criterium to use to update the model.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN
# x, y
# batch_size
# epochs
# how do I know I'm running estochastic gradient descent?
#classifier.fit(X_train, y_train, batch_size =  10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5) # transforms all values with regards to this condition.


#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000


new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



###################### Real accuracy  ############################
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifer():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
# make sure to use the latest keras api 
classifier = KerasClassifier(build_fn = build_classifer, batch_size =  10, epochs = 100)
accuracies = cross_val_score(estimator =  classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()


# Dropout regularization
# It's used to prvent overfitting => When we have large difference between training set and test set


################## Parameter Tunning ######################
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifer(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
# make sure to use the latest keras api 
classifier = KerasClassifier(build_fn = build_classifer)
parameters = { 'batch_size': [25, 32], 'epochs': [100, 500], 'optimizer': ['adam', 'rmsprop'] }

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_





