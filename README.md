# ST1-Capstone-Project-
ST1 Capstone Project

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#1.1 Load Data
file = '/Users/airajan/Desktop/Wholesale customers data (1).csv'
dataSet = pd.read_csv(file)

def overview (dataSet):
    return dataSet.head()

#1.1 Data set Information
def info(dataSet):
    dataSet.info()
    return dataSet.info()

#1.2 Check for missing values
def val(dataSet):
    dataSet.isnull().sum()
    return dataSet.isnull().sum()

#1.3 Statistical Info
def stat(dataSet):
    dataSet.describe()
    return dataSet.describe()

#2.1 and #2.2 Channel count and Region count
def chan (dataSet):
    # Data values
    dataSet["Channel"] = dataSet["Channel"].replace(1, "Hotel")
    dataSet["Channel"] = dataSet["Channel"].replace(2, "Retail")


    clmns = ['Channel']

    for cols in clmns:
        sns.set_style("darkgrid", {'grid.linestyle': ':'})
        plt.figure(figsize=(8, 5))
        sns.countplot(x=dataSet[cols], data=dataSet, palette='Set2')
        return plt.gcf()

def reg(dataSet):

    # Replace val
    dataSet["Region"] = dataSet["Region"].replace(1, "Lisbon")
    dataSet["Region"] = dataSet["Region"].replace(2, "Oporto")
    dataSet["Region"] = dataSet["Region"].replace(3, "Other")
    clmns2 = ['Region']
    for cols in clmns2:
        sns.set_style("darkgrid", {'grid.linestyle': ':'})
        plt.figure(figsize=(8, 5))
        sns.countplot(x=dataSet[cols], data=dataSet, palette='Set2')
        return plt.gcf()

# 2.3 Correlation of Region and Channel
def purch(dataSet):
    plt.figure(dpi=150)
    sns.pairplot(data=dataSet,hue='Region',palette='Set2')
    return plt.gcf()

def purch1(dataSet):
    plt.figure(dpi=150)
    sns.pairplot(data=dataSet,hue='Channel',palette='Set2')
    return plt.gcf()


#2.4 Product Purchases in unit
def howMuch(dataSet):
    plt.figure(figsize=(10, 8))
    sns.histplot(data=dataSet, x='Fresh', hue='Channel',  palette='Set3', multiple='stack');
    return plt.gcf()
def howMuch1(dataSet):
    plt.figure(figsize=(10, 8))
    sns.histplot(data=dataSet, x='Detergents_Paper', hue='Channel',  palette='Set3', multiple='stack');
    return plt.gcf()
def howMuch2(dataSet):
    plt.figure(figsize=(10, 8))
    sns.histplot(data=dataSet, x='Grocery', hue='Channel', palette='Set3', multiple='stack');
    return plt.gcf()

#2.5 Product Correlation
def prod(dataSet):
    plt.figure(figsize=(10, 8))
    sns.clustermap(dataSet.drop(['Region', 'Channel'], axis=1).corr(), annot=True);
    return plt.gcf()


#3.1
def pda(dataSet):
    plt.figure(figsize=(10, 8))
    dataSet.plot(kind='hist', alpha=0.8, bins=60, subplots=True, layout=(3, 2), legend=True, figsize=(12, 10))
    return plt.gcf()

#PDA

xAxis = dataSet.drop(['Channel', 'Region'], axis=1) # Drop the non-feature columns
yChan = dataSet['Channel'] #Use 'Channel' as the target variable for channel prediction
yReg = dataSet['Region'] #Use 'Region' as the target variable for region prediction



scaler = StandardScaler()
X_scaled = scaler.fit_transform(xAxis)


xTrain, xTest, yChanTrain, yChanTest = train_test_split(X_scaled, yChan, test_size=0.2, random_state=42)
xTrain, xTest, yRegTrain, yRegTest = train_test_split(X_scaled, yReg, test_size=0.2, random_state=42)

channelmod = LogisticRegression()
channelmod.fit(xTrain, yChanTrain)

regmod = LogisticRegression()
regmod.fit(xTrain, yRegTrain)

def chanMod(dataSet):
    pred1 = channelmod.predict(xTest)
    channel_accuracy = accuracy_score(yChanTest, pred1)
    return channel_accuracy

def regMod(dataSet):
    pred2 = regmod.predict(xTest)
    region_accuracy = accuracy_score(yRegTest, pred2)
    return region_accuracy

