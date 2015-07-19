## Approach

* Preprcessing
	** Trim white spaces and lowercase all column names

* EDA
	** Relationship between day of publishing and shares
	** Tried to create a feature like is_mon_or_sat (boolean) because they had unusually large values for shares
	** Relationship between shares and types of categories

* Feature Importance
	** Check p-value in the regression summary and remove those features which have a p-value greater than .05

* Feature Selection
	** Recursive Feature Elimination: Recursive eliminate features to find a subset of features which will contribute in creating final model
	** Applied Principal Component Analysis to reduce the dimensionality of the dataset.

* Also tried stacking but was not able to make it work on this dataset.

* Final Model: Ensemble of different models (Baseline model, one with categorical features, one with selective features, one on which PCA was applied)