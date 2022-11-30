with open("my_file_1.txt", "w") as f:
    f.write("sample content 1")

with open("my_file_1.txt", "a") as f:
    f.write("more content")
    
with open("my_file_1.txt", "w") as f:
    f.write("new content")
    
import numpy as np

sample_list = [10,20,30,40,50,60]

sameple_numpy_1d_array = np.array(sample_list)

sample_numpy_2d_array = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])

new_arr = sample_numpy_2d_array.reshape(2,6)

#create one row with max num of columns
new_arr2 = sample_numpy_2d_array.reshape(1,-1)

#create one column with max rows
new_arr3 = sample_numpy_2d_array.reshape(-1, 1)


new_sample = sample_numpy_2d_array[1:3, 2:4]
print(sample_numpy_2d_array.mean())

import pandas as pd

# gives elements a name
sample_series = pd.Series([10,20,30,40])
sample_series_2 = pd.Series([10,20,30,40], ['A', 'B', 'C', 'D'])

sample_dataframe = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
sample_dataframe_2 = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9], [10,11,12]],['row1', 'row2', 'row3', 'row4'], ['column1', 'column2', 'column3'])
print(sample_dataframe_2['column3'])
print(sample_dataframe_2.loc['row1'])
print(sample_dataframe_2.iloc[1:2, 2:4])

#Returns booleans of rows with values > 4 in column1
print(sample_dataframe_2['column1']>4)


print(sample_dataframe_2[sample_dataframe_2['column1']>4])

df = pd.read_csv("storepurchasedata.csv")