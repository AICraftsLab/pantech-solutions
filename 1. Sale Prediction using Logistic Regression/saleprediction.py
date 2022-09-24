#Importing Libraries
import pandas as pd #useful for loading dataset(as csv)
import numpy as np #to perform array

#select dataset file from local directory
#from google.colab import files
#uploaded = files.upload()

def press():
    input("Press any key to continue")

#load dataset
#dataset = pd.read_csv('dataset.csv')
dataset = pd.read_csv('C:/Abba/Coding/Courses/ML/myDataSet.csv')
print(dataset) #testing

press()

#summarize dataset
print(dataset.shape) #no. of rows and columns
print(dataset.head(5)) #top 5 lines

#segregate dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)
X = dataset.iloc[:, :-1].values #minus last column. And only the values
X

Y = dataset.iloc[:, -1].values #only last column. And only the values
Y

#splitting dataset into Train & Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

#feature scaling
    #We scale our data to make all the features contribute equally to the result
    #Fit_Transform - fit method is calculating the mean and variance of each of the features present in our data(the whole column, apply normal h(normalization))
    #Transform - Transform method is transforming all the features using the respective mean and variance,
    #We want our test data to be a completely new and a surprise set for our model
    #(No need for y bcoz it is a just 1 and 0, no difference in scale/unit/digit, but X has age in tens and salary in thousands i.e. there is large difference)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)

#training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() #loading the algorith
model.fit(X_train, y_train) #train(training has 75% of the data, trained and stored in this variable, use it for predicting)

#prediction for all test data
y_pred = model.predict(X_test) #(prediction for the remaining 25% of the data, maybe like the machine predict the status of the remaining data and compare his result with the actual result)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#evaluating model - Confusion matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score
#cm = confusion_matrix(y_test, y_pred)

#print("Confusion Matrix: ")
#print(cm)

print("Accuracy of the model: {0}%".format(accuracy_score(y_test, y_pred)*100))#compare the actual output(from the data) by the predicted output

press()

#predicting wheather a new customer with age & salary will buy or not
while 1:
    age = int(input("Enter New Customer's Age: "))
    sal = int(input("Enter New Customer's Salary: "))
    newCust = [[age,sal]]
    result = model.predict(sc.transform(newCust))#scaling should be same as X_test
    print(result)
    if result == 1:
        print("Customer will buy")
    else:
        print("Customer will not buy")
