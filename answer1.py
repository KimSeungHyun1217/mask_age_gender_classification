import shutil
import os
import pandas as pd

os.chdir('C:/Users/Hyun/Desktop')

data= pd.read_csv("examples.csv")

for i in data.columns:
  os.mkdir(i)


for i in data.name:
  src = 'data/' + i
  shutil.copy(src,'name')


for i in data['index']:
  src = 'data/' + i
  shutil.copy(src,'index')