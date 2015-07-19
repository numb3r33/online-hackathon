import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression


"""
Loads the csv file into a Pandas Dataframe
"""
def load_data(filename, index_col=None):
	if index_col:
		return pd.read_csv(filename, index_col=index_col)
	else:
		return pd.read_csv(filename)


"""
Returns target variable
"""

def get_target(df, target_col):
	return df[target_col]


"""
Trim column spaces
"""

def trim_column(df):
	return df.columns.map(lambda x: x.strip().lower())


"""
Returns floating and integer type features
"""

def float_int_features(df):
	columns = df.columns
	retCol = []

	for col in columns:
		if df[col].dtype == 'float64' or df[col].dtype == 'int64':
			if col != 'shares':
				retCol.append(col)

	return retCol

"""
Return features
"""

def return_features(df):
	columns = float_int_features(df)
	return df[columns]

"""
Add dummy encoding variables
"""
def add_dummy_variabes(features, data):
	features = pd.concat([features, pd.get_dummies(data.category_article, prefix='category_')], axis=1)
	return features


"""
Split into train and test data
"""

def split_data(X, y, test_size=0.2):
	return train_test_split(X, y, test_size=test_size, random_state=44)

"""
Normalize train features
"""

def normalize(X):
	scl = StandardScaler()
	return (scl.fit_transform(X), scl)

"""
Normalize test features
"""

def normalize_test(Xtest, scl):
	return scl.transform(Xtest)

"""
Build a gradient boosting regressor model
"""
def build_model(X, y):
	# gbr = GradientBoostingRegressor(learning_rate= 0.03, n_estimators=2000, max_depth=8, subsample=0.9)
	# rf = RandomForestRegressor(n_estimators=200)
	# lr = LinearRegression(fit_intercept=True)
	# knr = KNeighborsRegressor(n_neighbors=10, weights='uniform')
	# svr = SVR(C=5.0, kernel='linear')
	pls = PLSRegression(n_components=35)
	return pls.fit(X, y)

"""
Make predictions 
"""
def predict(model, Xtest):
	return model.predict(Xtest)

"""
RMSE score
"""

def RMSE(ytrue, ypred):
	return np.sqrt(mean_squared_error(ytrue, ypred))

"""
Create a submission file
"""
def make_submission(id, predictions, submissionFilename):
	submission = pd.DataFrame({"id": id, "predictions": predictions})
	submission.to_csv("./submissions/" + submissionFilename, index=False)