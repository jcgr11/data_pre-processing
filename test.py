# Print out the 5 most popular and commonly used Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

# Read in a dataset (in this case a CSV) 
data = pd.read_csv("data.csv")

# Start exploring the data to get some insights 
# See what columns each dataset has 
columns = data.columns

# Look at the first 5 lines of the dataset 
print(data.head())

# Summarize key statistics from the dataset 
print(data.describe())

# plot graphs to view the distribution of data 
sns.distplot(data['column1'], kde=False, bins=20)
sns.barplot(x="column2", y="column3", data=data)

# Use Seaborn to plot a correlation heatmap
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
import pandas as pd
#Read the csv file into a DataFrame
df = pd.read_csv('example_data.csv')

#Calculate the mean of each column
means = df.mean()

#Print out the mean of each column
for col in means.index:
    print("The mean of the '{}' column is {}".format(col, means[col]))
#Group and aggregate the data
grouped_data = df.groupby(['State', 'City']).agg({'Population':'sum', 'Average Temperature':'mean'})

#Print out the results
print(grouped_data)
