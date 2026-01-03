import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Reading .csv
dataset = pd.read_csv("Placement_BeginnerTask01.csv")

# Removing unnecessary columns
dataset = dataset.drop('StudentID', axis=1)

# Formatting data
dataset['ExtracurricularActivities'] = dataset['ExtracurricularActivities'].astype('category').cat.codes
dataset['PlacementTraining'] = dataset['PlacementTraining'].astype('category').cat.codes
dataset['PlacementStatus'] = dataset['PlacementStatus'].astype('category').cat.codes

# Checking the dataset
print(dataset.head())
print(dataset.describe())
print()

# X is the feature
X = dataset.iloc[:, :-1].values
# Y is the label
Y = dataset.iloc[:, -1].values

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Uses feature scaling to transform data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Classifies based on X_train and Y_train
clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000).fit(X_train, Y_train)

# Checks accuracy and a random value
print("Accuracy score =", clf.score(X_test, Y_test)*100, "%")
print()
print("Checking test subject:")
print("CGPA=8.1", "Internships=None", "Projects=1", "Workshops/Certifications=None", "AptitudeTestScore=81", "SoftSkillsRating = 4.2", "ExtracurricularActivities=No", "PlacementTraining=Yes", "SSC_Score=90", "HSC_Score=90", sep='\n')
prediction = clf.predict(scaler.transform([[8.1, 0, 1, 0, 81, 4.2, 0, 1, 90, 90]]))
if prediction[0] == 0:
    print("Unfortunately, you do not get placed.")
else:
    print("Congratulations! You got placed.")
print()

# Making confusion matrix and checking accuracy
Y_pred = clf.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print()
print("Accuracy score =", accuracy_score(Y_test, Y_pred))

for name, coefficient in zip(dataset.columns[:-1], clf.coef_[0]):
    print(f"{name}: {coefficient:.3f}")

# Now saving all the analytics

plt.figure(figsize=(6,4))
dataset['PlacementStatus'].value_counts().plot(kind='bar', color=['salmon', 'skyblue'])
plt.title('Placement Status Distribution')
plt.xlabel('Placement Status (0=Not Placed, 1=Placed)')
plt.ylabel('Number of Students')
plt.savefig("statistics/Placement_Status_Distribution.png", dpi=300, bbox_inches='tight')
plt.close()

keys = ['CGPA', 'AptitudeTestScore', 'SoftSkillsRating']
for key in keys:
    plt.hist(dataset[key], bins=10, color='lightgreen', edgecolor='black')
    plt.title(f'{key} Distribution')
    plt.xlabel(key)
    plt.ylabel('Number of Students')
    plt.savefig(f"statistics/{key}_Distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

plt.boxplot([dataset[dataset['PlacementStatus']==0]['CGPA'],
             dataset[dataset['PlacementStatus']==1]['CGPA']],
            tick_labels=['Not Placed', 'Placed'])
plt.ylabel('CGPA')
plt.title('CGPA vs Placement Status')
plt.savefig("statistics/CGPA_vs_Placement.png", dpi=300, bbox_inches='tight')
plt.close()

keys = ['Internships', 'Workshops/Certifications', 'ExtracurricularActivities', 'PlacementTraining']
for key in keys:
    pd.crosstab(dataset[key], dataset['PlacementStatus']).plot(kind='bar', stacked=True, color=['salmon','skyblue'])
    plt.title(f'{key} vs Placement Status')
    plt.xlabel(key)
    plt.ylabel('Count')
    key = key.replace('/', '_')
    plt.savefig(f"statistics/{key}_vs_Placement.png", dpi=300, bbox_inches='tight')
    plt.close()

placement = dataset.corr()['PlacementStatus'].sort_values(ascending=False)
placement = placement.drop('PlacementStatus')
plt.figure(figsize=(8,5))
placement.plot(kind='bar', color='skyblue')
plt.ylabel('Correlation with Placement Status')
plt.title('Feature Influence on Placement')
plt.savefig("statistics/Feature_Influence.png", dpi=300, bbox_inches='tight')
plt.close()