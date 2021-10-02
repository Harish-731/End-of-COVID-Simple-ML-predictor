import pandas as pd
see = pd.read_csv(r"D:\Pycharm projects\ML\lrdata.csv")
see.to_csv("lrdata")
print(pd.read_csv("lrdata"))