#Salary estimation of a job applicant wheather it will be greater or less than from previous company
#K value and mean error should both be low
#sklearn = scikit learn


#imports
import pandas as pd #for loading datasets
import numpy as np #to perform array

#Select dataset
from google.colab import files
uploaded = files.upload()

#load dataset
dataset = pd.read_csv('salary.csv') #load dataset into a variable
dataset

#summarize dataset (i.e. get num of rows and columns)
print(dataset.shape) #num of rows and cols
print(dataset.head(5)) #top 5 values

#mapping salary data to binay value (bcoz income column is not in pure numerical values, >50 = 1 and <=50 = 0)
income_set = set(dataset['income'])
dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
print(dataset.head(20))

#segregate dataset into X(input/independentVariable) & Y(output/dependentVariable)
X = dataset.iloc[:, :-1].values
X

Y = dataset.iloc[:, -1].values
Y

#Splitting dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0) #no randomization while split, it will start with first 75% for train and next 25% for test

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train

#finding best K-Value
error = []
from sklearn.neighbors import KNeighborsClassifier #algorithm
import matplotlib.pylot as plt #Data visualisation

#Calculating (mean) error for K Values btw 1 and 40
for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train) #(training)
    pred_i = model.predict(X_test) #(predicting)
    error.append(np.mean(pred_i != y_test)) #(calculating the error)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

#Training
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #(p is distance method, n_neighbors is the K Value Choosed)
model.fit(X_train, y_train)

#Predicting new Employee's salary
age = int(input("Enter New Employee's Age: "))
edu = int(input("Enter New Employee's Education: "))
cg = int(input("Enter New Employee's Capital Gain: "))
wh = int(input("Enter New Employee's Working Hours per week: "))
newEmp = [[age,edu,cg,wh]]
result = model.predict(sc.transform(newEmp))
print(result)

if result == 1:
    print("Employee might got Salary above 50K")
else:
    print("Employee might not got Salary above 50K")

#Predicion for all test data
y_pred = model.predict(X_test)

#Evaluating Model (checking accuracy)
from sklearn.metrics import accuracy_score #and confusion_matrix
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred)*100))

