#Support Vector Machine-Classifier (can be used as regression algo as well as classifier algo)
#All steps as the prev., new steps are data acquisation, data cleaning, data transformation
#The wider the Margin the better the algorithm
#google sklearn.svm


#(no need to import pandas bcoz we will use dataset from sklearn)
import numpy as np
from sklearn.datasets import load_digits #load dataset from sklearn

#Load Dataset(dataset to be loaded
    #classes 10 (i.e. 0-9)
    #samples per class approx. 180 (pics)
    #Samples total 1797
    #dimensionality 64 (8 rows x 8 cols)
    #features integers 0-16
dataset = load_digits()

#Summarize the data
print(dataset.data)  #(0-16, 0 = black, 16 = white)
print(dataset.target) #(output classes)

print(dataset.data.shape)
print(dataset.images.shape)

dataimageLength = len(dataset.images)
print(dataimageLength)

#Visualize the dataset
#n=10 #No. of sample out of samples total 1797 (to visualize, here it will visualize the 10th img)
n = int(input("Enter Sample Number, 0-1797\n"))

import matplotlib.pyplot as plt #data visualization
plt.gray()
plt.matshow(dataset.images[n])
plt.show()

dataset.images[n]

#Segregate Dataset into X(input/independent variable) and Y(output/dependent Var.)
X = dataset.images.reshape((dataimageLength,-1)) #(contains all imgs numerical data)
X
Y = dataset.target #(output classes, digits 0-9)
Y

#Splitting dataset into train & test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
print(X_train.shape)
print(X_test.shape)

#Trainig
from sklearn import svm
model = svm.SVC() #(gamma=0.001)
model.fit(X_train, y_train)

###

#Prediction for Test Data
y_pred = model.predict(X_test)

#Evaluate Model - Accuracy Score
from sklearn.metrics import accuracy_score
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred)*100))

#Play with Different Method(Tuning, i.e.changing some param)
#from sklearn import svm
model1 = svm.SVC(kernel='linear')
model3 = svm.SVC(gamma=0.001)
model4 = svm.SVC(gamma=0.001,C=0.1)

model1.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)

y_predModel1 = model1.predict(X_test)
y_predModel3 = model3.predict(X_test)
y_predModel4 = model4.predict(X_test)

print("Accuracy of the Model 1: {0}%".format(accuracy_score(y_test, y_predModel1)*100))
print("Accuracy of the Model 3: {0}%".format(accuracy_score(y_test, y_predModel3)*100))
print("Accuracy of the Model 4: {0}%".format(accuracy_score(y_test, y_predModel4)*100))

#Predicting what the digit is from Test Data
#n=199
int(input("Enter Sample Number, 0-1797\n"))
result = model.predict(dataset.images[n].reshape((1,-1)))
plt.imshow(dataset.images[n], cmap=plt.cm.gray_r, interpolation='nearest')
print(result)
print("\n")
plt.axis('off')
plt.title('%i' %result)
plt.show()

