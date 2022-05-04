#%% 
import pandas as pd 
import numpy as np
import altair as alt 
import seaborn as sns
#%%
df_train = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv")
df_test = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test.csv")
df_train.head(
)


#%%
features = df_train.filter([
    "education", "marital", "age", "housing" , 
    "loan", "day_of_week" , "campaign" , "y", 
    "job" , "default" , "pdays" , "previous"
]).sample(500)



# %%
def handle_graphs(featues):
    corr = features.drop(columns = 'y').corr()
    chart2 = sns.heatmap(corr)
    chart1 = sns.pairplot(features , hue='y')
    return chart1 and chart2

#%%
#Call the functions below 
handle_graphs(features)