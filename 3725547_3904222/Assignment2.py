# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Assignment 2
# Student ID: s3725547
# Student Name: Phil Steinke
# 
# Student ID: s3904222
# Student Name: Hayden Carr
# %% [markdown]
# ## Task 1: Retrieving and Preparing the Data (5%)
# - The student should retrieve the data correctly and prepare the data appropriately.
# Rubric: Data retrieved is completely free from errors and is in a form that is entirely appropriate for exploration

# %%
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sb

# %%
# import os
# os.getcwd()
data_file = '/Volumes/lenora/data-science/uni/2021/practicaldatascience/assignments/02assignment/3725547_3904222/data/Cervical Cancer Behavior Risk-DataSet.csv'
# data_file = 'C:/Users/Hayden/Desktop/RMIT/Practical Data Science with Python/practicaldatascienceass02-main/practicaldatascienceass02-main/3725547_3904222/Cervical Cancer Behavior Risk-DataSet.csv'
df = pd.read_csv(data_file)

df_cols = ['behavior_sexualRisk','behavior_eating','behavior_personalHygine','intention_aggregation','intention_commitment','attitude_consistency','attitude_spontaneity','norm_significantPerson','norm_fulfillment','perception_vulnerability','perception_severity','motivation_strength','motivation_willingness','socialSupport_emotionality','socialSupport_appreciation','socialSupport_instrumental','empowerment_knowledge','empowerment_abilities','empowerment_desires','ca_cervix']

# %%
# Evaluate the data
df.shape  # (72, 20)

# %%
df.head()

# %% [markdown]
# ## Task 2: Data Exploration (10%)

# %%
df.dtypes

# %%
# Null values
df.isnull().values.any()

# %%
df.describe().round(2)

# %% [markdown]
# ### 2.1 Explore each column ( or at least 10 columns if there are more than 10 columns), using appropriate descriptive statistics and graphs (if appropriate). For each explored column, please think carefully and report in your report in Task 4):
# Explore each column (or at least 10 columns if there are more than 10 columns),
# using appropriate descriptive statistics and graphs (if appropriate). Graphs should be formatted appropriately.
# Rubric: Descriptive statistics and/or graphs used are entirely appropriate and reflect the analyses for all (or at least 10) attributes. Graphs used are visualisation graphs, which are correctly formatted with all the necessary elements.
# Value counts of dataframe colums
# %% [markdown]
# #### 1) the way you used to explore a column(e.g. the graph);
# %% [markdown]
# #### 2) what you can observe from the way you used to explore it.
# (Please format each graph carefully, and use it in your final report. You need to include appropriate labels on the x-axis and y-axis, a title, and a legend. The fonts should be sized for good readability. Components of the graphs should be coloured appropriately, if applicable.)
# 
# Can probably kill this:
# TODO: appropriate labels on the x-axis and y-axis, a title, and a legend
# Null count of dataframe colums > 0
# alt.vconcat((chart1 | chart2 | chart3), (chart4 | chart5 | chart6))

# %%
alt.Chart(df, width=200, height=50).mark_bar().encode(
    x=alt.X(alt.repeat('repeat'), type='quantitative', bin=alt.Bin(maxbins=20)),
    y='count()'
    ).repeat(
        repeat=df_cols,
        columns=4
    )

# %% [markdown]
# ##### _fig 1: distribution of features_

# %% [markdown]
# ### Explore
# 2.2
# - Explore the relationship between all pairs of attributes (or at least 10 pairs of attributes, if there are more in the data), and
# - TODO: show the relationship in an appropriate graphs. You may choose which pairs of columns to focus on, but you need to generate a visualisation graph for each pair of attributes. Each of the attribute pair should address a plausible hypothesis for the data concerned.
# - TODO: In your report, for each plot (pair of attributes), state the hypothesis that you are investigating. Then, briefly discuss any interesting relationships (or lack of relationships) that you can observe from your visualisation.
# Please note you do not need to put all the graphs in your report, and you only need to include the representative ones and / or those showing significant information.
# 
# Rubric: Descriptive statistics and/or graphs used are entirely appropriate and reflect the analyses for all (or at least 10) pairs of attributes. Graphs used are visualisation graphs, which are correctly formatted with all the necessary elements. Hypotheses for each pair of data explored is concisely, clearly stated. Discussion of interesting relationships (or lack of relationships) between pairs of attributes is clear, comprehensive and insightful.

# %%
df_cols_compare_relationships = ['behavior_sexualRisk','intention_aggregation','intention_commitment','attitude_spontaneity','norm_significantPerson','perception_severity','motivation_strength','empowerment_knowledge']

alt.Chart(df).mark_rect().encode(
    alt.X(alt.repeat("column"), type = 'ordinal'),
    alt.Y(alt.repeat("row"), type='ordinal'),
    alt.Color('count()',
        scale=alt.Scale(scheme='blues')
    )
).repeat(
    row=['ca_cervix'],
    column=df_cols_compare_relationships
).properties(title = 'Cervical cancer vs All Other Variables')

# %% [markdown]
# ##### _fig 2: mark_rect plot of cervical cancer correlation with other variables FIXME_

# %%
# TODO: THIS NEEDS WORK:
alt.Chart(df).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color='ca_cervix:N'
).properties(
    width=150,
    height=150
).repeat(
    row=df_cols_compare_relationships,
    column=['ca_cervix']
).interactive()

# %% [markdown]
# ##### _fig 3: ca_cervix vs variable correlation_

# %% [markdown]
# Correlation plot

# %%
df_cols_compare_relationships2 = ['ca_cervix', 'behavior_sexualRisk', 'intention_aggregation', 'intention_commitment',
     'attitude_spontaneity', 'norm_significantPerson', 'perception_severity', 'motivation_strength', 'empowerment_knowledge']
df_cols_compare_relationships = df[df_cols_compare_relationships2]
# df_cols_compare_relationships.corr().loc['ca_cervix'].sort_values(ascending=False).head(8)

# %%
df.corr().loc['ca_cervix'].sort_values(ascending=True).head(12)

# %%
cols_high_corr=['ca_cervix','empowerment_abilities','perception_severity','empowerment_knowledge','motivation_strength','empowerment_desires','motivation_willingness','norm_fulfillment','perception_vulnerability','socialSupport_emotionality','behavior_personalHygine','socialSupport_appreciation','behavior_sexualRisk']
df_cols_high_corr = df[cols_high_corr]

# %% [markdown]
# ### Heatmap

# %%
dataplot = sb.heatmap(df_cols_high_corr.corr(),cmap='ocean')
plt.show()

# %% [markdown]
# ##### _fig 4: Correlation heatmap of variables_

# %% [markdown]
# ## Task 3: Data Modelling (15%)
# Rubric: A clear, comprehensive and robust description of steps with thorough and well-elaborated justifications for decisions made. The quality of the descriptions and justification will be consistent across both models. For both models, findings from training, evaluating and fine-tuning the models are clearly stated, well-developed and carefully interpreted. Results from comparison of models are logical, accurate, very insightful and support a clearly-articulated recommendation with a high degree of coherence and consistency.
# %% [markdown]
# ### Classification
# If you choose to do a Classification Task,
#  – Train and evaluate the model appropriately.
#  – Train the model by selecting the appropriate values for each parameter in the model. You need to show how you choose this values, and justify why you choose it.

# %%
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,auc,f1_score

# for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy import stats

# %% [markdown]
# # Model
# Model the data by treating it as either a Classification or Clustering Task, depending on your choice.
# You must use two different models(i.e. two Classification models, or two Clustering models), and when building each model, it must include the following steps:
# %% [markdown]
# ### • Select the appropriate features

# %%
target_col = 'ca_cervix'
dataML = df.drop(columns = target_col)
target = df[target_col]
dataML = preprocessing.MinMaxScaler().fit_transform(dataML)

# %% [markdown]
# ### Select the appropriate model(e.g. DecisionTree for classification) from sklearn.
# %% [markdown]
# cross-validation method

# %%
cv_method = RepeatedStratifiedKFold(n_splits=5, 
     n_repeats=3, 
     random_state=999)

# %% [markdown]
# ### Cross-validation method

# %%
dt_classifier = DecisionTreeClassifier(random_state=999)

params_DT = {'criterion': ['gini', 'entropy'],
             'max_depth': [1, 2, 3, 4, 5, 6],
             'min_samples_split': [2, 3]}

# %% [markdown]
# Grid search

# %%
gs_DT = GridSearchCV(dt_classifier, params_DT, cv=cv_method,verbose=1, scoring='accuracy', refit='accuracy')

gs_DT.fit(dataML, target)
results_DT = pd.DataFrame(gs_DT.cv_results_['params'])
results_DT['test_score'] = gs_DT.cv_results_['mean_test_score']
print(results_DT.columns)

# %% [markdown]
# Decision Tree Performance Comparison

# %%
alt.Chart(results_DT, 
          title='DT Performance Comparison'
         ).mark_line(point=True).encode(
    alt.X('max_depth', title='Maximum Depth'),
    alt.Y('test_score', title='Mean CV Score', aggregate='average', scale=alt.Scale(zero=False)),
    color='criterion'
)

# %% [markdown]
# ##### _fig 5: Decision Tree Performance comparison by mean CV score_

# %% [markdown]
# #### Decision Tree 50/50

# %%
X_train, X_test, y_train, y_test = train_test_split(
    dataML, target, test_size = 0.5, stratify=target, shuffle = True, random_state=999)

# %%
dt_classifier = gs_DT.best_estimator_
dt_classifier.fit(X_train, y_train)

# %% [markdown]
# #### Decision tree
# %% [markdown]
# Score

# %%
print(dt_classifier.score(X_test, y_test))

# %% [markdown]
# Confusion matrix

# %%
# this one was missing in mine, might be fine in your copy
dt_classifier = gs_DT.best_estimator_
dt_classifier.fit(X_train, y_train)

# %%
fit = dt_classifier.fit(X_train, y_train)
y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)
print(cm)

# %%
sb.heatmap(cm, annot=True,cmap='GnBu')
plt.show()

# %% [markdown]
# ##### _fig 6: heatmap of precision, recall, f1-score_

# %% [markdown]
# Precision, recall, F1-Score

# %%
print(classification_report(y_test,y_pre))

# %% [markdown]
# #### Decision Tree 60/40

# %%
X_train, X_test, y_train, y_test = train_test_split(
    dataML, target, test_size = 0.4, stratify=target, shuffle=True, random_state=999)

# %% [markdown]
# Decision tree

# %%
dt_classifier = gs_DT.best_estimator_
dt_classifier.fit(X_train, y_train)

# %% [markdown]
# DT Score

# %%
print(dt_classifier.score(X_test, y_test))

# %% [markdown]
# Confusion matrix

# %%
clf = KNeighborsClassifier (5,p=2)
fit = dt_classifier.fit(X_train, y_train)
y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)

# %% [markdown]
# DT Confusion Matrix

# %%
print(cm)

# %%
#finding it hard to find teal this could be an option, more colour https://matplotlib.org/stable/tutorials/colors/colormaps.html
sb.heatmap(cm, annot=True,cmap='GnBu')
plt.show()

# %% [markdown]
# ##### _fig 7: heatmap of precision, recall, f1-score_

# %% [markdown]
# Precision, recall, F1-Score

# %%
print(classification_report(y_test,y_pre))

# %% [markdown]
# 

# %%
X_train, X_test, y_train, y_test = train_test_split(
    dataML, target, test_size = 0.2, stratify=target, shuffle=True, random_state=999)

# %% [markdown]
# decision tree

# %%
dt_classifier = gs_DT.best_estimator_
dt_classifier.fit(X_train, y_train)

# %% [markdown]
# DT Score

# %%
print(dt_classifier.score(X_test, y_test))

# %% [markdown]
# Confusion matrix

# %%


# %%
fit = dt_classifier.fit(X_train, y_train)
y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)
# %% [markdown]
# DT Confusion Matrix

# %%
print(cm)

# %%
sb.heatmap(cm, annot=True,cmap='GnBu')
plt.show()

# %% [markdown]
# ##### _fig 8: heatmap of precision, recall, f1-score_

# %% [markdown]
# Precision, recall, F1-Score

# %%
print(classification_report(y_test,y_pre))

# %% [markdown]
# KNN hyperparameter tuning
# %% [markdown]
# cross-validation

# %%
cv_method = RepeatedStratifiedKFold(n_splits=5, 
     n_repeats=3, 
     random_state=999)

params_KNN = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
              'p': [1, 2, 5]}

gs_KNN = GridSearchCV(KNeighborsClassifier(), 
 params_KNN, 
 cv=cv_method,
 verbose=1, 
 scoring='accuracy', 
 refit='accuracy',
 return_train_score=True)

gs_KNN.fit(dataML, target)

print(gs_KNN.best_params_)

results_KNN = pd.DataFrame(gs_KNN.cv_results_['params'])

results_KNN['test_score'] = gs_KNN.cv_results_['mean_test_score']

results_KNN['metric'] = results_KNN['p'].replace([1,2,5], ["Manhattan", "Euclidean", "Minkowski"])
results_KNN

alt.Chart(results_KNN, 
          title='KNN Performance Comparison'
         ).mark_line(point=True).encode(
    alt.X('n_neighbors', title='Number of Neighbors'),
    alt.Y('test_score', title='Mean CV Score', scale=alt.Scale(zero=False)),
    color='metric'
)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    dataML, target, test_size = 0.5, stratify=target, shuffle=True, random_state=999)

# %% [markdown]
# Nearest neighbour score

# %%
knn_classifier = gs_KNN.best_estimator_
knn_classifier.fit(X_train, y_train)
print(knn_classifier.score(X_test, y_test))

# %% [markdown]
# Confusion matrix

# %%
y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)

# %% [markdown]
# NN Confusion Matrix

# %%
print(cm)

# %%
sb.heatmap(cm, annot=True,cmap='GnBu')
plt.show()

# %% [markdown]
# ##### _fig 9: heatmap of precision, recall, f1-score_

# %% [markdown]
# Precision, recall, F1-Score

# %%
print(classification_report(y_test,y_pre))

# %% [markdown]
# #### Nearest Neighbour 60/40

# %%
X_train, X_test, y_train, y_test = train_test_split(
    dataML, target, test_size = 0.4, stratify=target, shuffle=True, random_state=999)

# %% [markdown]
# Nearest neighbour score

# %%
knn_classifier = gs_KNN.best_estimator_
knn_classifier.fit(X_train, y_train)

# %% [markdown]
# NN Score

# %%
print(knn_classifier.score(X_test, y_test))

# %% [markdown]
# NN Confusion matrix

# %%
y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)
print(cm)

# %%
sb.heatmap(cm, annot=True,cmap='GnBu')
plt.show()

# %% [markdown]
# ##### _fig 10: heatmap of precision, recall, f1-score_

# %% [markdown]
# Precision, recall, F1-Score

# %%
print(classification_report(y_test,y_pre))

# %% [markdown]
# #### Nearest Neighbour 80/20

# %%
X_train, X_test, y_train, y_test = train_test_split(
    dataML, target, test_size = 0.2, stratify=target, shuffle=True, random_state=999)

# %% [markdown]
# Nearest neighbour score

# %%
knn_classifier = gs_KNN.best_estimator_
knn_classifier.fit(X_train, y_train)

# %% [markdown]
# NN Score

# %%
print(knn_classifier.score(X_test, y_test))

# %% [markdown]
# Confusion matrix

# %%
y_pre = fit.predict(X_test)
cm = confusion_matrix(y_test, y_pre)
print(cm)

# %%
sb.heatmap(cm, annot=True,cmap='GnBu')
plt.show()

# %% [markdown]
# ##### _fig 11: heatmap of precision, recall, f1-score_

# %% [markdown]
# Precision, recall, F1-Score

# %%
print(classification_report(y_test,y_pre))

# %%
cv_method_ttest = RepeatedStratifiedKFold(n_splits=5, 
 n_repeats=5, 
 random_state=999)

cv_results_KNN = cross_val_score(gs_KNN.best_estimator_,
     dataML,
     target, 
     cv=cv_method_ttest, 
     scoring='accuracy')

# %% [markdown]
# KNN Mean:

# %%
str(cv_results_KNN.mean().round(3))

cv_results_DT = cross_val_score(gs_DT.best_estimator_,
     dataML,
     target, 
     cv=cv_method_ttest, 
     scoring='accuracy')

# %% [markdown]
# DT Mean:

# %%
str(cv_results_DT.mean().round(3))

# %% [markdown]
# p-value:

# %%
str(stats.ttest_rel(cv_results_DT, cv_results_KNN).pvalue.round(3))

# %% [markdown]
# ### TODO: Compare the models
# After you have built two Classification models, or two Clustering models, on your data, the next step is to compare the models. You need to include the results of this comparison, including a recommendation of which model should be used, in your report(see next section).
# ---------------------------------------------------------- #
# %% [markdown]
# ## Task 4: Report (15%)
# Student ID: s3725547
# Student Name: Phil Steinke
# 
# Student ID: s3904222
# Student Name: Hayden Carr
# 
# We certify that this is all our own original work. If we took any parts from elsewhere, then they were non-essential parts of the assignment, and they are clearly attributed in our submission. We will show we agree to this honor code by typing "Yes": Yes.
# Please start your report since here ….
# ---
# Rubric: The report should be well-written and well-organised as required. It should include a clear description about the dataset and the research question you selected to solve. The process of modelling should be written clearly and completely, and the justification should be provided for how the features are selected, how the model is trained/evaluated, and and how the parameters are tuned.
# Rubric: The results from the two models should be compared, and please clearly describe what is your final recommendation to the problem you selected.
# Rubric: Report includes all the required elements Organisation of information is clear, logical, appropriately sequenced and consistent throughout Grammar and spelling mostly accurate.
