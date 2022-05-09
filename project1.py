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
from sklearn.feature_extraction.text import CountVectorizer
#%%
df_train = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv")
df_test = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test.csv")

df_train.loc[df_train["y"] == "no", "y"] = 0
df_train.loc[df_train["y"] == "yes", "y"] = 1

df_train.loc[df_train["housing"] == "no", "housing"] = 0
df_train.loc[df_train["housing"] == "yes", "housing"] = 1

df_train.loc[df_train["loan"] == "no", "loan"] = 0
df_train.loc[df_train["loan"] == "yes", "loan"] = 1
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



def explore_data(features):
    #null_data = df_train.isnull().sum()
    # unknown = df_train[features].str.find("unknown")
    #unknown = df_train.query([f"{feature} == 'unknown'"])
    x = df_train[df_train.eq("unknown").any(1)]
    print(x)



# %%
def handle_graphs(featues):
    corr = features.drop(columns = 'y').corr()
    chart2 = sns.heatmap(corr)
    chart1 = sns.pairplot(features , hue='y')
    return chart1 and chart2


#%%
# job , marital, education, housing, loan, contact, month, days_of_week , y
# Education: university.degree, high.school, basic.9y, professional.course, basic.4y, basic.6y, illiterate
# Contact: cellular, telephone
# Job: admin., blue-collar, technician, services, management, retired, entrepreneur, self-employed, housemaid, unemployed, student
# Month: mar, apr, may, jun. jul, aug, sep, oct, nov, dec  
# day_of_week: mon, tue, wed, thu, fri
#marrital_feature = features["marital"].str.contains('married', regex=True)
#features["marital"] = marrital_feature


X = features.drop(['y'] , axis=1)
y = features.filter(["y"] , axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% validatetion 

clf = RandomForestClassifier(random_state=0)
print(features.head())
clf.fit(X, y) 
y_pred=clf.predict(X_test)
#ValueError: could not convert string to float: 'married'
#%%
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#%%
#Call the functions below 
#handle_graphs(features)
#explore_data(features)