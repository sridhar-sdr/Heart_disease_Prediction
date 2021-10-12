
# coding: utf-8

# ## SimpleImputer
# ### This notebook outlines the usage of Simple Imputer (Univariate Imputation).
# ### Simple Imputer substitutes missing values statistics (mean, median, ...)
# #### Dataset: [https://github.com/subashgandyer/datasets/blob/main/heart_disease.csv]

# **Demographic**
# - Sex: male or female(Nominal)
# - Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
# 
# **Behavioral**
# - Current Smoker: whether or not the patient is a current smoker (Nominal)
# - Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.)
# 
# **Medical(history)**
# - BP Meds: whether or not the patient was on blood pressure medication (Nominal)
# - Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
# - Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
# - Diabetes: whether or not the patient had diabetes (Nominal)
# 
# **Medical(current)**
# - Tot Chol: total cholesterol level (Continuous)
# - Sys BP: systolic blood pressure (Continuous)
# - Dia BP: diastolic blood pressure (Continuous)
# - BMI: Body Mass Index (Continuous)
# - Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
# - Glucose: glucose level (Continuous)
# 
# **Predict variable (desired target)**
# - 10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)

# In[4]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[20]:


df=pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/heart_disease.csv")
df


# ### How many Categorical variables in the dataset?

# In[21]:


df.info()


# ### How many Missing values in the dataset?
# Hint: df.Series.isna( ).sum( )

# In[22]:


for i in range(len(df.columns)):
    missing_data = df[df.columns[i]].isna().sum()
    perc = missing_data / len(df) * 100
    print(f'Feature {i+1} >> Missing entries: {missing_data}  |  Percentage: {round(perc, 2)}')


# ### Bonus: Visual representation of missing values

# In[23]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# ### Import SimpleImputer

# In[24]:


from sklearn.impute import SimpleImputer


# ### Create SimpleImputer object with 'mean' strategy

# In[25]:


imputer= SimpleImputer(strategy='mean')


# ### Optional - converting df into numpy array (There is a way to directly impute from dataframe as well)

# In[26]:


data = df.values


# In[27]:


X = data[:, :-1]
y = data[:, -1]


# ### Fit the imputer model on dataset to calculate statistic for each column

# In[28]:


imputer.fit(X)


# ### Trained imputer model is applied to dataset to create a copy of dataset with all filled missing values from the calculated statistic using transform( ) 

# In[36]:


X_imputed = imputer.transform(X)


# ### Sanity Check: Whether missing values are filled or not

# In[32]:


sum(np.isnan(X).flatten())


# In[37]:


sum(np.isnan(X_imputed).flatten())


# In[38]:


plt.figure(figsize=(10,6))


# ### Let's try to visualize the missing values.

# In[39]:


plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)


# In[42]:


plt.figure(figsize=(10,6))
sns.heatmap(X_imputed.isna(), cbar=False, cmap='viridis', yticklabels=False)


# ### What's the issue here?
# #### Hint: Heatmap needs a DataFrame and not a Numpy Array

# In[43]:


df_imputed = pd.DataFrame(data=X_imputed)
df_imputed


# In[45]:


plt.figure(figsize=(10,6))
sns.heatmap(df_imputed.isna(), cbar=False, cmap='viridis', yticklabels=False)


# # Check if these datasets contain missing data
# ### Load the datasets

# In[46]:


X_train = pd.read_csv("X_train.csv")
Y_train = pd.read_csv("Y_train.csv")
Y_test = pd.read_csv("Y_test.csv")
X_test = pd.read_csv("X_test.csv")


# In[47]:


X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[48]:


plt.figure(figsize=(10,6))
sns.heatmap(X_train.isna(), cbar=False, cmap='viridis', yticklabels=False)


# ### Is there missing data in this dataset???

# In[ ]:


No Missing data


# # Build a Logistic Regression model Without imputation

# In[50]:


df=pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/heart_disease.csv")
X = df[df.columns[:-1]]
y = df[df.columns[-1]]


# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[52]:


model = LogisticRegression()


# In[53]:


model.fit(X,y)


# # Drop all rows with missing entries - Build a Logistic Regression model and benchmark the accuracy

# In[70]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


# In[61]:


df=pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/heart_disease.csv")
df


# In[63]:


df.shape


# ### Drop rows with missing values

# In[64]:


df=df.dropna()
df.shape


# ### Split dataset into X and y

# In[65]:


X= df[df.columns[:-1]]
X.shape


# In[67]:


y= df[df.columns[-1]]
y.shape


# ### Create a pipeline with model parameter

# In[69]:


model= LogisticRegression()


# In[72]:


pipe= Pipeline([("model",model)])


# ### Create a RepeatedStratifiedKFold with 10 splits and 3 repeats and random_state=1

# In[73]:


cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# ### Call cross_val_score with pipeline, X, y, accuracy metric and cv

# In[75]:


scores = cross_val_score(pipe, X,y, scoring= "accuracy", cv=cv, n_jobs=-1)


# In[81]:


scores


# In[82]:


scores


# ### Print the Mean Accuracy and Standard Deviation from scores

# In[80]:


print(f"Mean Accuracy: {round(np.mean(scores), 3)}  | Std: {round(np.std(scores), 3)}")


# 84.8% accuratewith +/- .5% tolerance

# # Build a Logistic Regression model with SimpleImputer Mean Strategy

# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


# In[85]:


df=pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/heart_disease.csv")
df


# ### Split dataset into X and y

# In[86]:


df.shape


# In[88]:


X= df[df.columns[:-1]]
X.shape


# In[89]:


y= df[df.columns[-1]]
y


# ### Create a SimpleImputer with mean strategy

# In[91]:


imputer= SimpleImputer(strategy= 'mean')


# ### Create a Logistic Regression model

# In[180]:


model=LogisticRegression()


# ### Create a pipeline with impute and model parameters

# In[93]:


pipe= Pipeline([('impute', imputer),('model',model)])


# ### Create a RepeatedStratifiedKFold with 10 splits and 3 repeats and random_state=1

# In[94]:


cv= RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# ### Call cross_val_score with pipeline, X, y, accuracy metric and cv

# In[95]:


scores= cross_val_score(pipe,X,y, scoring= 'accuracy', cv=cv,n_jobs=-1)


# In[96]:


scores


# ### Print the Mean Accuracy and Standard Deviation

# In[100]:


print(f"Mean Accuracy: {round(np.mean(scores), 3)}  | Std: {round(np.std(scores), 3)}")


# ### Which accuracy is better? 
# - Dropping missing values
# - SimpleImputer with Mean Strategy

# SimpleImputer with Mean Strategy

# # SimpleImputer Mean - Benchmark after Mean imputation with RandomForest

# ### Import libraries

# In[166]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


# ### Create a SimpleImputer with mean strategy

# In[167]:


imputer= SimpleImputer(strategy= 'median')


# ### Create a RandomForest model

# In[179]:


model1=RandomForestClassifier()


# ### Create a pipeline

# In[144]:


pipe= Pipeline([('impute', imputer),('model1',model)])


# ### Create RepeatedStratifiedKFold

# In[162]:


cv= RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# ### Create Cross_val_score

# In[163]:


rf_scores= cross_val_score(pipe,X,y, scoring= 'accuracy', cv=cv,n_jobs=-1)
rf_scores


# ### Print Mean Accuracy and Standard Deviation

# In[164]:


print(f"Median Accuracy: {round(np.mean(rf_scores), 3)}  | Std: {round(np.std(scores), 3)}")


# # Imputation with RandomForest

# In[181]:


#I have used the previous code script which taught in class by Prof.Vejey Subash Gandyer
results_rf =[]

strategies = ['mean', 'median', 'most_frequent','constant']

for s in strategies:
    pipeline = Pipeline([('impute', SimpleImputer(strategy=s)),('model1', model1)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    results_rf.append(scores)
    
for method, accuracy in zip(strategies, results_rf):
    print(f"Strategy: {method} >> Accuracy: {round(np.mean(accuracy), 3)}       |   Max accuracy: {round(np.max(accuracy), 3)}      |   Std: {round(np.std(scores), 3)}")
    


# # Imputation with KNN

# In[173]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier()


# In[209]:


results_knn =[]

strategies = ['mean', 'median', 'most_frequent','constant']

for s in strategies:
    pipeline = Pipeline([('impute', SimpleImputer(strategy=s)),('model2', model2)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    results_knn.append(scores)
    
for method, accuracy in zip(strategies, results_knn):
    
    print(f"Strategy: {method} >> Accuracy: {round(np.mean(accuracy), 3)}       |   Max accuracy: {round(np.max(accuracy), 3)}      |   Std: {round(np.std(scores), 3)}")


# # Imputation With LogisticRegression

# In[ ]:


model=LogisticRegression()


# In[178]:


results =[]

strategies = ['mean', 'median', 'most_frequent','constant']

for s in strategies:
    pipeline = Pipeline([('impute', SimpleImputer(strategy=s)),('model', model)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    results.append(scores)
    
for method, accuracy in zip(strategies, results):
    print(f"Strategy: {method} >> Accuracy: {round(np.mean(accuracy), 3)}       |   Max accuracy: {round(np.max(accuracy), 3)}      |   Std: {round(np.std(scores), 3)}")
    


# # Imputation With SVM

# In[186]:


from sklearn import svm
model3= svm.SVC()


# In[187]:


results_svm =[]

strategies = ['mean', 'median', 'most_frequent','constant']

for s in strategies:
    pipeline = Pipeline([('impute', SimpleImputer(strategy=s)),('model3', model3)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    results_svm.append(scores)
    
for method, accuracy in zip(strategies, results_svm):
    print(f"Strategy: {method} >> Accuracy: {round(np.mean(accuracy), 3)}       |   Max accuracy: {round(np.max(accuracy), 3)}      |   Std: {round(np.std(scores), 3)}")


# # Imputation With NaiveBayes

# In[206]:


from sklearn.naive_bayes import GaussianNB
model4 = GaussianNB()


# In[208]:


results_nb =[]

strategies = ['mean', 'median', 'most_frequent','constant']

for s in strategies:
    pipeline = Pipeline([('impute', SimpleImputer(strategy=s)),('model4', model4)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    results_nb.append(scores)
    
for method, accuracy in zip(strategies, results_nb):
    print(f"Strategy: {method} >> Accuracy: {round(np.mean(accuracy), 3)}       |   Max accuracy: {round(np.max(accuracy), 3)}      |   Std: {round(np.std(scores), 3)}")


# # Imputation With DecisionTree

# In[201]:



from sklearn.tree import DecisionTreeClassifier
model5 = DecisionTreeClassifier(random_state=0)


# In[203]:


results_dt =[]

strategies = ['mean', 'median', 'most_frequent','constant']

for s in strategies:
    pipeline = Pipeline([('impute', SimpleImputer(strategy=s)),('model5', model5)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    results_dt.append(scores)
    
for method, accuracy in zip(strategies, results_dt):
    print(f"Strategy: {method} >> Accuracy: {round(np.mean(accuracy), 3)}       |   Max accuracy: {round(np.max(accuracy), 3)}      |   Std: {round(np.std(scores), 3)}")
    


# # Assignment
# # Run experiments with different Strategies and different algorithms
# 
# ## STRATEGIES
# - Mean
# - Median
# - Most_frequent
# - Constant
# 
# ## ALGORITHMS
# - Logistic Regression
# - KNN
# - Random Forest
# - SVM
# - Any other algorithm of your choice

# #### Hint: Collect the pipeline creation, KFold, and Cross_Val_Score inside a for loop and iterate over different strategies in a list and different algorithms in a list

# # Q1: Which is the best strategy for this dataset using Random Forest algorithm?
# - MEAN
# - MEDIAN
# - MOST_FREQUENT
# - CONSTANT
                             Answer : MEAN  Accuracy: 0.849       |   Max accuracy: 0.868      |   Std: 0.006
# # Q2:  Which is the best algorithm for this dataset using Mean Strategy?
# - Logistic Regression
# - Random Forest
# - KNN
# - any other algorithm of your choice (BONUS)
   For Mean Strategy  Answer: 1.KNN             Accuracy: 0.837       |   Max accuracy: 0.858      |   Std: 0.01 (Less Variance)
                              2.SVM             Accuracy: 0.85        |   Max accuracy: 0.856      |   Std: 0.005
                    
# # Q3: Which is the best combination of algorithm and best Imputation Strategy overall?
# - Mean , Median, Most_frequent, Constant
# - Logistic Regression, Random Forest, KNN

# In[ ]:


1.KNN (Mean)                   Accuracy: 0.837       |   Max accuracy: 0.858      |   Std: 0.01 (Less Variance)
2.SVM  (Mean)                  Accuracy: 0.85        |   Max accuracy: 0.856      |   Std: 0.005
3.RandomForest(most_frequent)  Accuracy: 0.849       |   Max accuracy: 0.863      |   Std: 0.006   
            

