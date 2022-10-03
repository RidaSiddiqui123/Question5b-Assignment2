#-------------------------------------------------------------------------
# AUTHOR: Rida Siddiqui
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2 - Question 5b
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
import csv
from sklearn.naive_bayes import GaussianNB

db = []

#reading the training data in a csv file
#--> add your Python code here
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)


#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here

Outlook = {
    "Sunny": 1,
    "Overcast": 2,
    "Rain": 3
    }
Temperature = {
    "Cool": 1,
    "Mild": 2,
    "Hot": 3
}
Humidity = {
    "Normal": 1,
    "High": 2
}
Wind = {
    "Strong": 1,
    "Weak": 2
}
X = []
Y = []
for row in db:
    X.append([Outlook[row[1]], Temperature[row[2]], Humidity[row[3]], Wind[row[4]]])

# X =
#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =

label = {
    "Yes": 1,
    "No": 2
}
for row in db:
    Y.append(label[row[5]])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

db1 = []
#reading the test data in a csv file
#--> add your Python code here
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db1.append (row)


#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

test_data = []
for row in db1:
    test_data.append([Outlook[row[1]], Temperature[row[2]], Humidity[row[3]], Wind[row[4]]])


for i in range(len(test_data)):

    arr = clf.predict_proba([test_data[i]])[0]

    maximum_value = max(arr)
    if maximum_value >= 0.75:
        if maximum_value == arr[0]:
            print(db1[i][0].ljust(15) + db1[i][1].ljust(15) + db1[i][2].ljust(15) + db1[i][3].ljust(15) + db1[i][4].ljust(
                15) + "Yes".ljust(15) + str(round(maximum_value, 2)).ljust(15))

        else:
            print(db1[i][0].ljust(15) + db1[i][1].ljust(15) + db1[i][2].ljust(15) + db1[i][3].ljust(15) + db1[i][4].ljust(
                    15) + "No".ljust(15) + str(round(maximum_value, 2)).ljust(15))
