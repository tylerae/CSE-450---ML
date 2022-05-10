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