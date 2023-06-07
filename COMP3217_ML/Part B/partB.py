#Taken from Scikit

from sklearn import datasets, neighbors, linear_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import torch
import csv

# import some data
data = pd.read_csv('C:/Users/11391/Desktop/cyber physical/cw2/TrainingDataMulti.csv',header=None)
test_data = pd.read_csv('C:/Users/11391/Desktop/cyber physical/cw2/TestingDataMulti.csv',header=None)
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values
test_features = test_data.values
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

""" #print shape of the array for X and Y. Also get value of targets
print (features.shape)
print (labels)

# Converts the training and testing data to PyTorch tensors
features_train = torch.tensor(features_train, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.float32)
features_test = torch.tensor(features_test, dtype=torch.float32)
labels_test= torch.tensor(labels_test, dtype=torch.float32)

print(features_train)
print(labels_train)
print(features_test)
print(labels_test)

# Count the number of samples for each label in the training and testing sets
test_label_counts = torch.unique(labels_test, return_counts=True)
train_label_counts = torch.unique(labels_train, return_counts=True)

print("Test Label Counts:")
for label, count in zip(test_label_counts[0], test_label_counts[1]):
    print(f"Label {label}: {count}")

print("\nTrain Label Counts:")
for label, count in zip(train_label_counts[0], train_label_counts[1]):

    print(f"Label {label}: {count}") """
"""  # Find the best n_estimators for rfc
cross = []
for i  in range(0,300,10):
    rfc = RandomForestClassifier(n_estimators=i+1, random_state=0)
    cross_score = cross_val_score(rfc, features_train, labels_train, cv=5).mean()
    cross.append(cross_score)
plt.plot(range(1,301,10),cross)
plt.xlabel('n_estimators')
plt.ylabel('acc')
plt.show()
print((cross.index(max(cross))*10)+1,max(cross)) """

# Set rfc and logistic and compare them
rfc = RandomForestClassifier(n_estimators=271,random_state=0)

print(
    "RFC score: %f"
    % rfc.fit(features_train, labels_train).score(features_test, labels_test))

# Choose rfc
predictions = rfc.predict(features_test)


# Calculate classification metrics
accuracy = accuracy_score(labels_test, predictions)
precision = precision_score(labels_test, predictions, average='weighted')
recall = recall_score(labels_test, predictions, average='weighted')
f1 = f1_score(labels_test, predictions, average='macro')

print("Classification Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


#Get confusion matrix
cm = confusion_matrix(labels_test, predictions, labels=rfc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
disp.plot()
plt.show()

#Get results on actual test labels and predicted labels
test_predictions = rfc.predict(test_features)

#print (test_features)
print (test_predictions)
results=pd.DataFrame(test_features)
results.insert(128, 'labels', test_predictions) 
results.to_csv('TestingResultsMulti.csv',header=False,mode='a',index=False)

