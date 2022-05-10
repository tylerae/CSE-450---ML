#%% 
from multiprocessing import dummy
import pandas as pd 
import numpy as np
import altair as alt 
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score


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

features = df_train.filter([ "age", "housing" , 
    "loan", "campaign" , "y" ,
     "pdays" , "previous"
]).sample(500)

df_train.loc[df_train["y"] == "no", "y"] = 0
df_train.loc[df_train["y"] == "yes", "y"] = 1

df_train.loc[df_train["housing"] == "no", "housing"] = 0
df_train.loc[df_train["housing"] == "yes", "housing"] = 1

df_train.loc[df_train["loan"] == "no", "loan"] = 0
df_train.loc[df_train["loan"] == "yes", "loan"] = 1


df_train.dropna()

#%%


X_pred = df_train.filter([ "age", "housing" , 
    "loan", "campaign" , 
     "pdays" , "previous"])


y_pred = df_train.filter(["y"])
y_pred=y_pred.astype('int')

X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .34, 
    random_state = 76) 

model = RandomForestClassifier()
model = model.fit(X_train, y_train)

predict = model.predict(X_test)
show = mean_absolute_error(y_test,predict)
print(show)
print(confusion_matrix(y_test, predict))
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))