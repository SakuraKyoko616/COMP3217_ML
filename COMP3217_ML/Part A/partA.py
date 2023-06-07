#Taken from Scikit

from sklearn import datasets, neighbors, linear_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import torch
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# import  data
data = pd.read_csv('C:/Users/11391/Desktop/cyber physical/cw2/TrainingDataBinary.csv',header=None)
test_data = pd.read_csv('C:/Users/11391/Desktop/cyber physical/cw2/TestingDataBinary.csv',header=None)
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values
test_features = test_data.values
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#print shape of the array for X and Y. Also get value of targets
print (features.shape)
print (labels)

""" # Converts the training and testing data to PyTorch tensors
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
    
# Logistic regression classifier
logistic = linear_model.LogisticRegression(max_iter=5000,C=0.1)

""" # Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = PCA()

# Define a Standard Scaler to normalize inputs
scaler = StandardScaler()

# Parameters of pipelines can be set using '__' separated parameter names:
pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logistic)])
param_grid = {
    "pca__n_components": [5, 10, 30, 45, 60],
    "logistic__C": np.logspace(-1, 1, 1),
}
search = GridSearchCV(pipe, param_grid, n_jobs=2,cv=5)
search.fit(features_train, labels_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# Plot the PCA spectrum
pca.fit(features_train)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(
    np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
)
ax0.set_ylabel("PCA explained variance ratio")

ax0.axvline(
    search.best_estimator_.named_steps["pca"].n_components,
    linestyle=":",
    label="n_components chosen",
)
ax0.legend(prop=dict(size=12))

# For each number of components, find the best classifier results
results = pd.DataFrame(search.cv_results_)
print (results)
components_col = "param_pca__n_components"
best_clfs = results.groupby(components_col).apply(
    lambda g: g.nlargest(1, "mean_test_score")
)

best_clfs.plot(
    x=components_col, y="mean_test_score", yerr="std_test_score", legend=False, ax=ax1
)
ax1.set_ylabel("Classification accuracy (val)")
ax1.set_xlabel("n_components")

plt.xlim(-1, 70)

plt.tight_layout()
plt.show()
 """
print(
    "LogisticRegression score: %f"
    % logistic.fit(features_train, labels_train).score(features_test, labels_test))

predictions = logistic.predict(features_test)


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
cm = confusion_matrix(labels_test, predictions, labels=logistic.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logistic.classes_)
disp.plot()
plt.show()

#Get results on actual test labels and predicted labels
test_predictions = logistic.predict(test_features)

#print (test_features)
print (test_predictions)
results=pd.DataFrame(test_features)
results.insert(128, 'labels', test_predictions) 
results.to_csv('TestingResultsBinary.csv',header=False,mode='a',index=False)

