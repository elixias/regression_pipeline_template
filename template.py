import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

dataset = pd.read_csv("train.csv")
dataset = dataset.dropna()
#dataset = dataset.dropna(subset=["columns",""...])

#target = dataset.iloc[:,-1].ravel()
target = dataset["Survived"].values
#features = dataset.iloc[:,:-1].copy()
features = dataset.iloc[:,[2,4,5,9,11]].copy()

#categorical data
#features['column'].fillna(features['column'].value_counts().index[0], inplace=True)

#filling in na values for categories. for numerical dtypes, use imputer
#features['column'].fillna(features['column'].value_counts().index[0], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0)

numerical_features = features.dtypes == 'float'
categorical_features = ~numerical_features

preprocess = make_column_transformer(
    (make_pipeline(SimpleImputer(), StandardScaler()),numerical_features),
    (OneHotEncoder(), categorical_features)
)

model = make_pipeline(
    preprocess,
    LogisticRegression(solver="lbfgs", multi_class="auto")
)

model.fit(X_train, y_train)
print("logistic regression score: %f" % model.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV
param_grid = {
    'columntransformer__pipeline__simpleimputer__strategy': ['mean', 'median'],
    'logisticregression__C': [0.1, 1.0, 1.0],
    }
	
grid_clf = GridSearchCV(model, param_grid, cv=10, iid=False)
grid_clf.fit(X_train, y_train);
grid_clf.best_params_

print("best logistic regression from grid search: %f" % grid_clf.best_estimator_.score(X_test, y_test))

#backward elimination
"""
fittrain = preprocess.fit_transform(X_train);
print(fittrain)

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y_train, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(fittrain.values, SL)
print(X_Modeled)
"""