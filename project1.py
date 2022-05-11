#%% 
from multiprocessing import dummy
import pandas as pd 
import numpy as np
import altair as alt 
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
#%%
df_train = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv")
df_test = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test.csv")
alt.data_transformers.enable('json')

df_train.dropna(axis=0)
#%%

def filter_unknown(col_name):
  unknown_ser = df_train[df_train[col_name] == "unknown"]
  #only take out the one's where the target variable is no
  #remove = unknown_ser[unknown_ser['y'] == "no"]
  return df_train.drop(unknown_ser.index)

col_list = [
    "education", "marital", "age", "housing" , 
    "loan", "day_of_week" , "campaign" , "y", 
    "job" , "default", "pdays", "previous"
]

for c in col_list:
    before = len(df_train[df_train[c] == "unknown"])
    # Remove unknown rows
    df_train = filter_unknown(c)
    after = len(df_train[df_train[c] == "unknown"])
    print(before - after, "rows removed for column:", c)
#%%
features = df_train.filter([ "marital" ,"education" , "age", "housing" , 
    "loan", "campaign" , "y" , "day_of_week" , "job", "default"
     "pdays" , "previous"
])

# df_train.loc[df_train["marital"] == "married", "y"] = 0
# df_train.loc[df_train["y"] == "yes", "y"] = 1


#job
df_train.job[df_train.job == "housemaid"] = 1
df_train.job[df_train.job == "services"] = 2
df_train.job[df_train.job == "admin."] = 3
df_train.job[df_train.job == "blue-collar"] = 4
df_train.job[df_train.job == "technician"] = 5
df_train.job[df_train.job == "retired"] = 6
df_train.job[df_train.job == "management"] = 7
df_train.job[df_train.job == "unemployed"] = 8 
df_train.job[df_train.job == "self-employed"] = 9
df_train.job[df_train.job == "entrepreneur"] = 10
df_train.job[df_train.job == "student"] = 11
#marital
df_train.marital[df_train.marital == "married"] = 1
df_train.marital[df_train.marital == "single"] =2
df_train.marital[df_train.marital == "divorced"] = 3
#education
df_train.education[df_train.education == "basic.4y"] = 1
df_train.education[df_train.education == "high.school"] = 2
df_train.education[df_train.education == "basic.6y"] = 3
df_train.education[df_train.education == "basic.9y"] = 4
df_train.education[df_train.education == "professional.course"] = 5
df_train.education[df_train.education == "university.degree"] = 6
df_train.education[df_train.education == "illiterate"] = 7
#default
df_train.default[df_train.default == "yes"] = 1
df_train.default[df_train.default == "no"] = 0
#housing
df_train.housing[df_train.housing == 'yes'] = 1
df_train.housing[df_train.housing == "no"] = 0
#loan
df_train.loan[df_train.loan == 'yes'] = 1
df_train.loan[df_train.loan == 'no'] = 0
#contact 
df_train.contact[df_train.contact == 'telephone'] = 1
df_train.contact[df_train.contact == 'cellular'] = 0
#month
df_train.month[df_train.month == 'may'] = 5
df_train.month[df_train.month == 'jun'] = 6
df_train.month[df_train.month == 'jul'] = 7
df_train.month[df_train.month == 'aug'] = 8
df_train.month[df_train.month == 'sep'] = 9
df_train.month[df_train.month == 'oct'] = 10
df_train.month[df_train.month == 'nov'] = 11
df_train.month[df_train.month == 'dec'] = 12
df_train.month[df_train.month == 'mar'] = 3
df_train.month[df_train.month == 'apr'] = 4
#day_of_week
df_train.day_of_week[df_train.day_of_week == 'mon'] = 1
df_train.day_of_week[df_train.day_of_week == 'tue'] = 2
df_train.day_of_week[df_train.day_of_week == 'wed'] = 3
df_train.day_of_week[df_train.day_of_week == 'thu'] = 4
df_train.day_of_week[df_train.day_of_week == 'fri'] = 5
#poutcome
df_train.poutcome[df_train.poutcome == 'nonexistent'] = 1
df_train.poutcome[df_train.poutcome == 'failure'] = 2
df_train.poutcome[df_train.poutcome == 'success'] = 3
#y
df_train.y[df_train.y == 'yes'] = 1
df_train.y[df_train.y == 'no'] = 0

# jeremys stuff

for c in col_list:
    before = len(df_test[df_test[c] == "unknown"])
    # Remove unknown rows
    df_test = filter_unknown(c)
    after = len(df_test[df_test[c] == "unknown"])
    print(before - after, "rows removed for column:", c)
#%%
features = df_test.filter([ "marital" ,"education" , "age", "housing" , 
    "loan", "campaign" , "y" , "day_of_week" , "job", "default"
     "pdays" , "previous"
])

# df_train.loc[df_train["marital"] == "married", "y"] = 0
# df_train.loc[df_train["y"] == "yes", "y"] = 1


#job
df_test.job[df_test.job == "housemaid"] = 1
df_test.job[df_test.job == "services"] = 2
df_test.job[df_test.job == "admin."] = 3
df_test.job[df_test.job == "blue-collar"] = 4
df_test.job[df_test.job == "technician"] = 5
df_test.job[df_test.job == "retired"] = 6
df_test.job[df_test.job == "management"] = 7
df_test.job[df_test.job == "unemployed"] = 8 
df_test.job[df_test.job == "self-employed"] = 9
df_test.job[df_test.job == "entrepreneur"] = 10
df_test.job[df_test.job == "student"] = 11
#marital
df_test.marital[df_test.marital == "married"] = 1
df_test.marital[df_test.marital == "single"] =2
df_test.marital[df_test.marital == "divorced"] = 3
#education
df_test.education[df_test.education == "basic.4y"] = 1
df_test.education[df_test.education == "high.school"] = 2
df_test.education[df_test.education == "basic.6y"] = 3
df_test.education[df_test.education == "basic.9y"] = 4
df_test.education[df_test.education == "professional.course"] = 5
df_test.education[df_test.education == "university.degree"] = 6
df_test.education[df_test.education == "illiterate"] = 7
#default
df_test.default[df_test.default == "yes"] = 1
df_test.default[df_test.default == "no"] = 0
#housing
df_test.housing[df_test.housing == 'yes'] = 1
df_test.housing[df_test.housing == "no"] = 0
#loan
df_test.loan[df_test.loan == 'yes'] = 1
df_test.loan[df_test.loan == 'no'] = 0
#contact 
df_test.contact[df_test.contact == 'telephone'] = 1
df_test.contact[df_test.contact == 'cellular'] = 0
#month
df_test.month[df_test.month == 'may'] = 5
df_test.month[df_test.month == 'jun'] = 6
df_test.month[df_test.month == 'jul'] = 7
df_test.month[df_test.month == 'aug'] = 8
df_test.month[df_test.month == 'sep'] = 9
df_test.month[df_test.month == 'oct'] = 10
df_test.month[df_test.month == 'nov'] = 11
df_test.month[df_test.month == 'dec'] = 12
df_test.month[df_test.month == 'mar'] = 3
df_test.month[df_test.month == 'apr'] = 4
#day_of_week
df_test.day_of_week[df_test.day_of_week == 'mon'] = 1
df_test.day_of_week[df_test.day_of_week == 'tue'] = 2
df_test.day_of_week[df_test.day_of_week == 'wed'] = 3
df_test.day_of_week[df_test.day_of_week == 'thu'] = 4
df_test.day_of_week[df_test.day_of_week == 'fri'] = 5
#poutcome
df_test.poutcome[df_test.poutcome == 'nonexistent'] = 1
df_test.poutcome[df_test.poutcome == 'failure'] = 2
df_test.poutcome[df_test.poutcome == 'success'] = 3
#y
df_test.y[df_test.y == 'yes'] = None
df_test.y[df_test.y == 'no'] = None


# add a y column that will be blank
#df_train.dropna()
df_test.head()
#df_train.dropna()
df_train.head()
#%%

X_train = df_train.filter([ "marital" ,"education" , "age", "housing" , 
    "loan", "campaign" , "day_of_week" , "job", "default"
     "pdays" , "previous"
])

y_train = df_train.filter(["y"])
y_train= y_train.astype('int')

X_test = df_test.filter([ "marital" ,"education" , "age", "housing" , 
    "loan", "campaign" , "day_of_week" , "job", "default"
     "pdays" , "previous"
])

y_test = df_test.filter(["y"])
y_test= y_test.astype('int')


#X_train, X_test, y_train, y_test = train_test_split(
#   X_pred, 
#   y_pred, 
#   test_size = .30, 
#    random_state = 76) 

model = RandomForestClassifier()
model = model.fit(X_train, y_train.values.ravel())

predict = model.predict(X_test)
show = mean_absolute_error(y_test,predict)
print(show)
print(confusion_matrix(y_test, predict))
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))
print(accuracy_score(y_test, predict))

model2 = DecisionTreeClassifier(max_depth=2)
model2 = model2.fit(X_train, y_train)


tree.plot_tree(model2, fontsize=10)



# saving the predictions to a csv
y_test.to_csv('predictions.csv')  