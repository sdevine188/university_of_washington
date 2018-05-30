import pandas as pd
import numpy as np
import os as os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR
from copy import deepcopy


##################################################################

# summary: this script imports the adult dataset from the UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Adult
# the data is from census, with variables for demographic, social, education, etc for individuals
# the purpose is to predict whether the individual makes over $50k per year
#
# the script cleans the data, replaces outliers with the mean, 
# replaces missing numeric values with the mean, replaces missing categorical variables with the mode,
# converts categorical variables into dummy variables,
# and conducts minmax normailization
#
# the script then creates a decision tree model and a logistic regression model,
# evaluates the performance metrics for both models,
# and concludes by comparing the performance metrics and the relative 
# pros and cons of each model


########################################################


#os.getcwd()
#os.chdir("H:/python/university_of_washington/data_process_and_tools/lesson_4")

# load adult data
# load adult dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
adult_raw = pd.read_csv(url, header = None)
#adult_raw = pd.read_csv("adult.csv", header = None)
adult = adult_raw

# inspect adult
adult.shape
adult.head()
adult.columns

# add variable names
var_names = ["age", "workclass", "fnlwgt", "education", "education_number", 
                 "marital_status", "occupation", "relationship", "race", "sex", 
                 "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]

adult.columns = var_names

# check var_names
adult.columns
adult.head()
adult.dtypes


#############################################################


# replace outliers
# create function to replace outliers with the mean for numeric variables
def replace_outliers_w_mean(dataframe, variable):
        variable_mean = dataframe[variable].mean()
        variable_std = dataframe[variable].std()
        upper_limit = variable_mean + 2 * variable_std
        lower_limit = variable_mean - 2 * variable_std
#        return(dataframe.loc[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit), variable])
        dataframe.loc[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit), variable] = variable_mean
        return(dataframe)
        
# find numeric variables
adult.select_dtypes(include = ["number"]).columns      

# replace outliers with mean for numeric variables

# age
# not going to replace "outliers" for age
adult.age.dtype.name
adult.age.value_counts()
adult.age.isnull().sum()
adult.age.describe()
adult[["age"]].sort_values(by = ["age"], ascending = False).head()

# fnlwgt
adult.fnlwgt.dtype.name
adult.fnlwgt.value_counts()
adult[["fnlwgt"]].isnull().sum()
adult[["fnlwgt"]].describe()
adult[["fnlwgt"]].sort_values(by = ["fnlwgt"], ascending = False).head()
replace_outliers_w_mean(adult, "fnlwgt")
adult[["fnlwgt"]].sort_values(by = ["fnlwgt"], ascending = False).head()

# education_number
# not going to remove "outliers" for education
adult.education_number.dtype.name
adult.education_number.value_counts()
adult[["education_number"]].isnull().sum()
adult[["education_number"]].describe()
adult[["education_number"]].sort_values(by = ["education_number"], ascending = False).head()

# capital_gain
adult.capital_gain.dtype.name
adult.capital_gain.value_counts()
adult.groupby(["capital_gain"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["capital_gain"]].isnull().sum()
adult[["capital_gain"]].describe()
adult[["capital_gain"]].sort_values(by = ["capital_gain"], ascending = False).head()
adult.capital_gain.mean()
replace_outliers_w_mean(adult, "capital_gain")
adult[["capital_gain"]].sort_values(by = ["capital_gain"], ascending = False).head()

# capital_loss
adult.capital_loss.dtype.name
adult.capital_loss.value_counts()
adult.groupby(["capital_loss"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["capital_loss"]].isnull().sum()
adult[["capital_loss"]].describe()
adult[["capital_loss"]].sort_values(by = ["capital_loss"], ascending = False).head()
replace_outliers_w_mean(adult, "capital_loss")
adult[["capital_loss"]].sort_values(by = ["capital_loss"], ascending = False).head()

# hours_per_week
adult.hours_per_week.dtype.name
adult.hours_per_week.value_counts()
adult.groupby(["hours_per_week"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["hours_per_week"]].isnull().sum()
adult[["hours_per_week"]].describe()
adult[["hours_per_week"]].sort_values(by = ["hours_per_week"], ascending = False).head()
replace_outliers_w_mean(adult, "hours_per_week")
adult[["hours_per_week"]].sort_values(by = ["hours_per_week"], ascending = False).head()



#################################################################


# clean string variables by imputing missing value placeholders with the mode value
adult.dtypes

# workclass
adult.workclass.dtype.name
adult.workclass.value_counts()
adult.workclass.values
adult.loc[:, "workclass"] = adult.workclass.str.strip()
adult.groupby(["workclass"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["workclass"]].isnull().sum()
adult[["workclass"]].describe()

# get mode value for imputation
workclass_mode = adult.groupby(["workclass"]).size().reset_index(name = "n").\
        sort_values("n", ascending = False).iloc[0:1, ]["workclass"]
workclass_mode

# impute
adult.loc[adult.workclass == " ?", "workclass"] = workclass_mode
adult.workclass.value_counts()


####################


# education
# no need for imputation/cleaning
adult.education.dtype.name
adult.education.values
adult.loc[:, "education"] = adult.education.str.strip()
adult.education.value_counts()
adult.groupby(["education"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["education"]].isnull().sum()
adult[["education"]].describe()


########################


# marital_status
# no need for cleaning/imputation
adult.marital_status.dtype.name
adult.marital_status.values
adult.loc[:, "marital_status"] = adult.marital_status.str.strip()
adult.marital_status.value_counts()
adult.groupby(["marital_status"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["marital_status"]].isnull().sum()
adult[["marital_status"]].describe()


#######################


# occupation
adult.occupation.dtype.name
adult.occupation.values
adult.loc[:, "occupation"] = adult.occupation.str.strip()
adult.occupation.value_counts()
adult.groupby(["occupation"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["occupation"]].isnull().sum()
adult[["occupation"]].describe()

# get mode value for imputation
occupation_mode = adult.groupby(["occupation"]).size().reset_index(name = "n").\
        sort_values("n", ascending = False).iloc[0:1, ]["occupation"]
occupation_mode

# impute
adult.loc[adult.occupation == " ?", "occupation"] = occupation_mode
adult.occupation.value_counts()


###################


# relationship
# no need for cleaning/imputation
adult.relationship.dtype.name
adult.relationship.values
adult.loc[:, "relationship"] = adult.relationship.str.strip()
adult.relationship.value_counts()
adult.groupby(["relationship"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["relationship"]].isnull().sum()
adult[["relationship"]].describe()


####################


# race
# no need for cleaning/imputation
adult.race.dtype.name
adult.race.values
adult.loc[:, "race"] = adult.race.str.strip()
adult.race.value_counts()
adult.groupby(["race"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["race"]].isnull().sum()
adult[["race"]].describe()


######################


# sex
# no need for cleaning/imputation
adult.sex.dtype.name
adult.sex.values
adult.loc[:, "sex"] = adult.sex.str.strip()
adult.sex.value_counts()
adult.groupby(["sex"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["sex"]].isnull().sum()
adult[["sex"]].describe()


######################


# native_country
adult.native_country.dtype.name
adult.native_country.values
adult.loc[:, "native_country"] = adult.native_country.str.strip()
adult.native_country.value_counts()
adult.groupby(["native_country"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["native_country"]].isnull().sum()
adult[["native_country"]].describe()

# get mode value for imputation
native_country_mode = adult.groupby(["native_country"]).size().reset_index(name = "n").\
        sort_values("n", ascending = False).iloc[0:1, ]["native_country"]
native_country_mode

# impute
adult.loc[adult.native_country == " ?", "native_country"] = native_country_mode
adult.native_country.value_counts()


######################


# income
# no need for cleaning/imputation
adult.income.dtype.name
adult.income.values
adult.loc[:, "income"] = adult.income.str.strip()
adult.income.value_counts()
adult.groupby(["income"]).size().reset_index(name = "n").\
        sort_values(by = ["n"], ascending = False).head()
adult[["income"]].isnull().sum()
adult[["income"]].describe()

# replace income with dummy
adult.loc[(adult.income == "<=50K"), "income"] = 0
adult.loc[(adult.income == ">50K"), "income"] = 1
adult.income.value_counts()
adult.loc[:, "income"] = pd.to_numeric(adult.income)
adult.income.dtype.name


##################################################


# convert categorical variables to dummies
adult.dtypes

# get categorical variables
adult_categorical_vars = adult.select_dtypes(include = ["object"])
adult_categorical_vars.columns

# get dummies
adult_dummies = pd.get_dummies(data = adult_categorical_vars)
adult_dummies
adult_dummies.columns

# confirm that dummies are still in same order as original df
adult_dummies["workclass_Private"][0:10]
adult["workclass"][0:10]

# get numeric variables from original df, which will then be binded with adult_dummies
adult_numeric_vars = adult.loc[:, ~(adult.columns.isin(adult_categorical_vars.columns))]
adult_numeric_vars.columns

# drop outcome variable income, since we don't need dummies for it
adult_numeric_vars = adult_numeric_vars.drop(["income"], axis = 1)
adult_numeric_vars.columns

# bind adult_numeric_vars with adult_dummies
adult2 = pd.concat([adult.income, adult_numeric_vars, adult_dummies], axis = 1)
adult2.columns
adult2.head()


##################################################


# create normalize function
def normalize(X): # max-min normalizing
    N, m = X.shape
    Y = np.zeros([N, m])
    
    for i in range(m):
#        mX = min(X[:,i])
        mX = min(X.iloc[:, i])
#        Y[:,i] = (X[:,i] - mX) / (max(X[:,i]) - mX)
        Y[:,i] = (X.iloc[:,i] - mX) / (max(X.iloc[:,i]) - mX)

    
    return Y


#################################################################3
    

# normalize data
adult2.columns
adult2.dtypes

adult_norm = normalize(adult2)
type(adult_norm)
adult_norm = pd.DataFrame(adult_norm)
adult_norm.head()
adult_norm.columns = adult2.columns
adult_norm.head()


#################################################################


# save adult_norm to file
#os.getcwd()
#os.chdir('H:\\python\\university_of_washington\\data_process_and_tools\\lesson_8')
#adult_norm.to_csv("adult_normalized.csv", index = False)


################################################################33
    

# create split_dataset function
def split_dataset(data, r): # split a dataset in matrix format, using a given ratio for the testing set
	N = len(data)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

	n = int(round(N*r)) # number of elements in testing sample
	nt = N - n # number of elements in training sample
	ind = -np.ones(n,int) # indexes for testing sample
	R = np.random.randint(N) # some random index from the whole dataset
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) # ensure that the random index hasn't been used before
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) # remaining indexes	
	X = data[ind_,1:] # training features
	XX = data[ind,1:] # testing features
	Y = data[ind_,0] # training targets
	YY = data[ind,0] # testing targests
	return X, XX, Y, YY


######################################################3


# get np.array of adult_norm to use in split_dataset
adult_norm_array = adult_norm.values
type(adult_norm_array)

# split into train/test data
r = 0.2
random.seed(123)
X, XX, Y, YY = split_dataset(adult_norm_array, r)

# save arrays as dataframes
adult_train = pd.DataFrame(X)
adult_train.columns = adult_norm.columns[1:]
adult_train.columns
adult_train.head()
adult_train.shape

adult_train_income = pd.DataFrame(Y)
adult_train_income.columns = ["income"]
adult_train_income.columns
adult_train_income.head()
adult_train_income.shape

adult_test = pd.DataFrame(XX)
adult_test.columns = adult_norm.columns[1:]
adult_test.columns
adult_test.head()
adult_test.shape

adult_test_income = pd.DataFrame(YY)
adult_test_income.columns = ["income"]
adult_test_income.columns
adult_test_income.head()
adult_test_income.shape


################################################################
################################################################
################################################################


# train decision tree
# the decision tree was chosen because it's a relatively easily interpreted classifier
tree_model = tree.DecisionTreeClassifier(min_samples_leaf = 50) # default parameters are fine
random.seed(123)
tree_model.fit(adult_train, adult_train_income)

tree_model.tree_.node_count
tree_model.classes_

# get predicted class on test data
adult_test_pred = tree_model.predict(adult_test)
type(adult_test_pred)
adult_test_pred

# get predicted probabilities on test data
adult_test_pred_prob = tree_model.predict_proba(adult_test)
type(adult_test_pred)
adult_test_pred_prob = pd.DataFrame(adult_test_pred_prob)
adult_test_pred_prob.columns = ["class_negative", "class_positive"]
adult_test_pred_prob.head()

# add pred and pred_prob to adult_test
adult_test = adult_test.assign(pred = adult_test_pred)
adult_test = adult_test.assign(pred_prob = adult_test_pred_prob.class_positive)
adult_test.columns
adult_test.head()


################################################################


# confusion matrix and other metrics comparing actual outcome to predicted outcome
confusion_matrix(y_true = adult_test_income, y_pred = adult_test.pred)
print(classification_report(y_true = adult_test_income, y_pred = adult_test.pred))

accuracy_score(y_true = adult_test_income, y_pred = adult_test.pred)
precision_score(y_true = adult_test_income, y_pred = adult_test.pred)
recall_score(y_true = adult_test_income, y_pred = adult_test.pred)
f1_score(y_true = adult_test_income, y_pred = adult_test.pred)


################################################################


# roc and auc
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

# get roc curve stats
fpr, tpr, th = roc_curve(y_true = adult_test_income, y_score = adult_test.pred_prob) # False Positive Rate, True Posisive Rate, probability thresholds
fpr
tpr
th

# inspect fpr, tpr, and th in dataframe
fpr_df = pd.DataFrame(fpr)
tpr_df = pd.DataFrame(tpr)
th_df = pd.DataFrame(th)

roc_curve_df = pd.concat([fpr_df, tpr_df, th_df], axis = 1)
roc_curve_df.columns = ["fpr", "tpr", "th"]
roc_curve_df.sort_values("fpr").head()
roc_curve_df.head()

# get auc
AUC = auc(fpr, tpr)
AUC

# print output
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))

#####################


# plot roc curve
plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()


################################################################
################################################################
################################################################


# create adult_test2 for use with logistic regression model, 
# don't want to overwrite the adult_test data used for the decision tree model
adult_test2 = pd.DataFrame(XX)
adult_test2.columns = adult_norm.columns[1:]
adult_test2.columns
adult_test2.head()
adult_test2.shape


##################################################################


# train logistic regression model
# a logistic regression model was chosen because it's very commonly used and 
# yields significance stats on the coefficients
logistic_model = LogisticRegression()
random.seed(123)
logistic_model.fit(adult_train, adult_train_income)

logistic_model.coef_

# get predicted class on test data
adult_test_pred = logistic_model.predict(adult_test2)
type(adult_test_pred)
adult_test_pred

# get predicted probabilities on test data
adult_test_pred_prob = logistic_model.predict_proba(adult_test2)
type(adult_test_pred)
adult_test_pred_prob = pd.DataFrame(adult_test_pred_prob)
adult_test_pred_prob.columns = ["class_negative", "class_positive"]
adult_test_pred_prob.head()

# add pred and pred_prob to adult_test
adult_test2 = adult_test2.assign(pred = adult_test_pred)
adult_test2 = adult_test2.assign(pred_prob = adult_test_pred_prob.class_positive)
adult_test2.columns
adult_test2.head()


################################################################


# confusion matrix and other metrics comparing actual outcome to predicted outcome
confusion_matrix(y_true = adult_test_income, y_pred = adult_test2.pred)
print(classification_report(y_true = adult_test_income, y_pred = adult_test2.pred))

accuracy_score(y_true = adult_test_income, y_pred = adult_test2.pred)
precision_score(y_true = adult_test_income, y_pred = adult_test2.pred)
recall_score(y_true = adult_test_income, y_pred = adult_test2.pred)
f1_score(y_true = adult_test_income, y_pred = adult_test2.pred)


################################################################


# roc and auc
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

# get roc curve stats
fpr, tpr, th = roc_curve(y_true = adult_test_income, y_score = adult_test2.pred_prob) # False Positive Rate, True Posisive Rate, probability thresholds
fpr
tpr
th

# inspect fpr, tpr, and th in dataframe
fpr_df = pd.DataFrame(fpr)
tpr_df = pd.DataFrame(tpr)
th_df = pd.DataFrame(th)

roc_curve_df = pd.concat([fpr_df, tpr_df, th_df], axis = 1)
roc_curve_df.columns = ["fpr", "tpr", "th"]
roc_curve_df.sort_values("fpr").head()
roc_curve_df.head()

# get auc
AUC2 = auc(fpr, tpr)
AUC2

# print output
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))

#####################


# plot roc curve
plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()


#####################################################################
####################################################################
######################################################################


# compare the decision tree and logistic models
# the decision tree model had AUC = .89, compared to logistic model AUC = .90
# the decision tree model had accuracy = .847, compared to logistic model accuracy = .849
# the decision tree model had precision = .73, compared to logistic model precision = .74
# the decision tree model had recall = .57, compared to logistic model recall = .57
# the decision tree model had f1 score = .65, compared to logistic model f1 score = .65

# overall the models fit well and are very close in predictive performance on the test set
# the logistic model has very slighly higher AUC, accuracy, and precision
# depending on how the model might be used, we could choose between the two models
# based on how easily interpreted they are.  the decision tree is easy to interpret, but
# time consuming since it has so many nodes.  the logistic model is a bit more 
# challenging to interpret, but it's relatively simple in that is has  
# coefficients for each variable along with significance statistics.

