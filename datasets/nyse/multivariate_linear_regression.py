import numpy as np;
import pandas as pd;

#based on the results of linear regression we do not thing this is the best model to model stock prices

def linear_regression(X,y):
		"""Performs a linear regression of the data passed through X and 
		the output y using the normal equation(X'X)^-1*X'*y)"""
		
		transpose_X=np.transpose(X);# implements the formula (X'*X)^-1*X'*y 
		X=np.matmul(transpose_X,X);
		X=np.linalg.inv(X);
		X=np.matmul(X,transpose_X);
		X=np.matmul(X,y);
		
		return X;

df=pd.read_csv("OP_NYSE.csv");
df.insert(0,"x0",1);

#Removes all non numeric columns from the data frame after importing the data
df=df.loc[:, df.columns != 'Unnamed: 0.1'];
df=df.loc[:, df.columns != 'Ticker Symbol'];
df=df.loc[:, df.columns != 'Period Ending'];
df=df.loc[:, df.columns != 'GICS Sector'];
df=df.loc[:, df.columns != 'GICS Sub Industry'];

df=df.dropna(axis='rows'); #drops all rows without full aet of data may need to revise in the future.


open_array=df["Open"].values #find the values in of the pricing categories
close_array=df["Close"].values
low_array=df["Low"].values;
high_array=df["High"].values;

df=df.loc[:, df.columns != 'Open']; #Gets rid of the price values
df=df.loc[:, df.columns != 'Close'];
df=df.loc[:, df.columns != 'Low'];
df=df.loc[:, df.columns != 'High'];

X=df.values;



average_array=np.add(open_array,close_array); #adds the array together and takes their average
average_array=np.add(average_array,low_array);
average_array=np.add(average_array,high_array);
average_array=average_array/4;
#average_array=np.transpose(average_array);

coefficents=linear_regression(X,average_array); #Utiizes the linear_regression function to find coefficents of the linear regression

for i in range(len(X)): #Prints value in comparison to actual value in the data
	print(np.matmul(coefficents,np.transpose(X[i])),average_array[i]);