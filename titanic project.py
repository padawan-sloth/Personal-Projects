import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Load data
train_data = pd.read_csv('/path/to/train.csv')
test_data = pd.read_csv('/path/to/test.csv')

# Preprocess data
## Fill missing values
imputer_age = SimpleImputer(strategy="median")
train_data['Age'] = imputer_age.fit_transform(train_data[['Age']])
test_data['Age'] = imputer_age.transform(test_data[['Age']])

imputer_fare = SimpleImputer(strategy="median")
train_data['Fare'] = imputer_fare.fit_transform(train_data[['Fare']])
test_data['Fare'] = imputer_fare.transform(test_data[['Fare']])

## Encode categorical data
label_encoders = {}
for column in ['Sex', 'Cabin', 'Embarked']:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column].astype(str))
    test_data[column] = le.transform(test_data[column].astype(str))
    label_encoders[column] = le

## Feature engineering
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Model training
X_train = train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)
y_train = train_data['Survived']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting
X_test = test_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
submission.to_csv('/path/to/submission.csv', index=False)
