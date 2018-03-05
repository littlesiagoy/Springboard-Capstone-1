import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor


'''Data Cleansing '''
#######################################################################################################################
#######################################################################################################################

''' Import/Clean the poverty dataset. '''
#######################################################################################################################
poverty_dataset = pd.read_excel('poverty_rate_dataset.xls', skiprows=[0, 1, 2, 3], header=0)
poverty_dataset = poverty_dataset[['STATE', 'Total', 'Number', 'Percent']]

# This will give us the rows where each year starts.
test = poverty_dataset[poverty_dataset['Total'].isnull()]

# This is the list of years we want to pull out.
list_of_years = [1980, 1985, 1990, 1995, 2000, 2005, 2010]

# year index + 2, next year is the formula for the indices in the poverty dataset
data_1980 = poverty_dataset.iloc[1961:2012]
data_1985 = poverty_dataset.iloc[1696:1747]
data_1990 = poverty_dataset.iloc[1431:1482]
data_1995 = poverty_dataset.iloc[1166:1217]
data_2000 = poverty_dataset.iloc[901:952]
data_2005 = poverty_dataset.iloc[636:687]
data_2010 = poverty_dataset.iloc[371:422]

# Add a year column to the data.
data_sets = [data_1980, data_1985, data_1990, data_1995, data_2000, data_2005, data_2010]
years = [1980, 1985, 1990, 1995, 2000, 2005, 2010]

for index in range(len(data_sets)):
    data_sets[index]['Year'] = years[index]

# The result is the cleansed poverty_dataset.
final_poverty_dataset = pd.concat(data_sets)

# The columns are STATE, Total, Number, Percent, Year
# Convert the total number, and percent columns to floats.
final_poverty_dataset.Total = final_poverty_dataset.Total.apply(pd.to_numeric)
final_poverty_dataset.Number = final_poverty_dataset.Number.apply(pd.to_numeric)
final_poverty_dataset.Percent = final_poverty_dataset.Percent.apply(pd.to_numeric)

''' Import/Clean the Cardio Dataset. '''
########################################################################################################################
# Import the cardio datset.
cardio_dataset = pd.read_excel('cardiovascular_dataset.xlsx', sheet_name='Cardiovascular diseases', skiprows=[0])

# Drop the 2014 column and the # change in mortality rate column.
cardio_dataset = cardio_dataset.drop(['Mortality Rate, 2014*', '% Change in Mortality Rate, 1980-2014'], axis=1)

# After inspecting the codebook, we see that all the states will be given an unique FIPS value.  It ranges from 1 to 56.
# Initialize empty dataset.
final_cardio_dataset = pd.DataFrame()

# Return the row for each state and append it to the empty dataset.
for index in range(57):
    final_cardio_dataset = final_cardio_dataset.append(cardio_dataset[cardio_dataset.FIPS == index])

final_cardio_dataset = pd.melt(final_cardio_dataset, id_vars=['Location', 'FIPS'])

final_cardio_dataset.columns = [['Location', 'FIPS', 'Year', 'Mortality Rate']]

# Replace the values in the variable column
final_cardio_dataset.replace('^Mortality Rate, ', '', regex=True, inplace=True)
final_cardio_dataset.replace('\*$', '', regex=True, inplace=True)

# Get rid of the confidence interval in the mortality rate column.
final_cardio_dataset['Mortality Rate'] = final_cardio_dataset['Mortality Rate'].replace(' .*', '', regex=True)

# Convert all the columns except for the states into numeric.
final_cardio_dataset.Year = final_cardio_dataset.Year.apply(pd.to_numeric)
final_cardio_dataset['Mortality Rate'] = final_cardio_dataset['Mortality Rate'].apply(pd.to_numeric)

# Replace all District of Columbia values in cardio dataset to D.C.
final_cardio_dataset.replace('District of Columbia', 'D.C.', regex=True, inplace=True)

final_cardio_dataset.drop('FIPS', axis=1, inplace=True)

# Rename the final poverty dataset columns.
final_poverty_dataset.columns = ['STATE', 'Total (1000s)', 'Number (1000s)', 'Percent', 'Year']

''' Import/Clean the US GDP dataset. '''
#######################################################################################################################
us_gdp = pd.read_csv('US GDP 1980-2017.csv', skiprows=[0, 1, 2, 3], index_col=0)

# Pull out only the US GDP for each year.
us_gdp = us_gdp.drop([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])

# Turn from the wide format to short format.
us_gdp = pd.melt(us_gdp)

# rename columns and drop the first row.
us_gdp.columns = ['Year', 'Gross Domestic Product ($)']
us_gdp = us_gdp.drop([0])

# Convert column from str to int.
us_gdp['Year'] = us_gdp['Year'].astype(int)

# Set the index so we can grab the years we want.
us_gdp = us_gdp.set_index('Year')
us_gdp = us_gdp.loc[years]

# Multiply the GDp to change the scale from billions of dollars to just dollars.
us_gdp.loc[:, 'Gross Domestic Product ($)'] = us_gdp.loc[:, 'Gross Domestic Product ($)'] * 1000000000

''' Import/Clean the US Productivity dataset. '''
#######################################################################################################################
# Import the dataset.
productivity = pd.read_csv('US Overall Labour Productivity 1970-2016.csv')

# We want the measures of productivity to be in USD.
productivity = productivity.loc[productivity.loc[:, 'MEASURE'] == 'USD']

# We only want the Time and value columns.
productivity = productivity[['TIME', 'Value']]

# Rename the columns to make it more clear.
productivity.columns = ['Year', 'GDP Per Hour Worked']

productivity = productivity.set_index('Year')

# We want specific set of years.
productivity = productivity.loc[years]

''' Import/Clean the State Unemployment Rate dataset. '''
#######################################################################################################################
state_unemploymnet = pd.read_excel('US State Unemployment Rates 1980-2015.xls', sheet_name='States', skiprows=[0, 1,
                                                                                                               2, 3, 4])

# Get rid of the unncessary column.
state_unemploymnet = state_unemploymnet.drop('Fips', axis=1)

# Get melt the data from wide to long.
state_unemploymnet = pd.melt(state_unemploymnet, id_vars=['Area'])

# Replace the District of Columbia value to D.C.
state_unemploymnet.replace('District of Columbia', 'D.C.', regex=True, inplace=True)

state_unemploymnet.columns = ['STATE', 'Year', 'Unemployment Rate']

''' Import/Clean the Corruption Dataset. '''
#######################################################################################################################
# Import the data.
corruption = pd.read_excel('final-transparency-indices-scores-september-2014.xlsx', sheet_name='ATI, ITI AND TI SCORES')

# Pull out the United States data.
corruption = corruption.loc[corruption.loc[:, 'Country'] == 'United States']

# Pull out only the years we want.
corruption = corruption.set_index('Year')
corruption = corruption.loc[years]

# Drop the extra columns.
corruption = corruption[['Information Transparency Score', 'Accountability Transparency Score',
                         'Transparency Index Score']]

''' Combine the datasets. '''
#######################################################################################################################
# Combine the overall US data.
us_dataset = us_gdp.join([corruption, productivity])

# Create a State column with just the name United States.
us_dataset['STATE'] = 'United States'

# Combine the state level data.
states_dataset = state_unemploymnet.merge(final_poverty_dataset)

# Change the final cardio dataset columns.
final_cardio_dataset.columns = ['STATE', 'Year', 'Mortality Rate']

# Merge the cardio dataset with the state dataset.
states_dataset = final_cardio_dataset.merge(states_dataset)

''' Impute the overall US level data into the state dataset. '''
########################################################################################################################
states_dataset['Gross Domestic Product ($)'] = ''
states_dataset['Information Transparency Score'] = ''
states_dataset['Accountability Transparency Score'] = ''
states_dataset['Transparency Index Score'] = ''
states_dataset['GDP Per Hour Worked'] = ''

for value in years:
    states_dataset.loc[states_dataset.loc[:, 'Year'] == value, 'Gross Domestic Product ($)'] = us_dataset.loc[
        value, 'Gross Domestic Product ($)']
    states_dataset.loc[states_dataset.loc[:, 'Year'] == value, 'Information Transparency Score'] = us_dataset.loc[
        value, 'Information Transparency Score']
    states_dataset.loc[states_dataset.loc[:, 'Year'] == value, 'Accountability Transparency Score'] = us_dataset.loc[
        value, 'Accountability Transparency Score']
    states_dataset.loc[states_dataset.loc[:, 'Year'] == value, 'Transparency Index Score'] = us_dataset.loc[
        value, 'Transparency Index Score']
    states_dataset.loc[states_dataset.loc[:, 'Year'] == value, 'GDP Per Hour Worked'] = us_dataset.loc[
        value, 'GDP Per Hour Worked']

# The state dataset is now the final master dataset.
master_dataset = states_dataset

master_dataset.to_excel('master_dataset.xlsx', index=False)

''' Smoothing out the data due to different unit of measurements. '''
########################################################################################################################
# Due to the different units of measure between the variables, preprocessing is needed.

# Take out our target variable of mortality rates before we perform the preprocessing.
y = master_dataset['Mortality Rate']

# Take out the STATE, year, and mortality Rate variables for ease of interpretation after preprocessing.
samples = master_dataset.drop(['STATE', 'Year', 'Mortality Rate'], axis=1)

# Scale the with the RobustScaler, since the number of outliers is unknown.
data = RobustScaler().fit_transform(samples)
data = pd.DataFrame(data, columns=samples.columns)

# Add the state and year data back.
data['STATE'] = master_dataset['STATE']
data['Year'] = master_dataset['Year']

# Reorgnize the columns.
data = data[['STATE', 'Year', 'Unemployment Rate', 'Total (1000s)', 'Number (1000s)', 'Percent',
             'Gross Domestic Product ($)', 'Information Transparency Score', 'Accountability Transparency Score',
             'Transparency Index Score', 'GDP Per Hour Worked']]

# Change the value of the state into a number.
data_set = sorted(set(data.STATE))
for index, value in enumerate(data_set):
    data.loc[data.loc[:, 'STATE'] == value, 'STATE'] = index

''' Exploratory Data Analysis '''
#######################################################################################################################
#######################################################################################################################

''' Outlier detection with LOF '''
########################################################################################################################
# Initialize the model.
lof = LocalOutlierFactor()

# Calculate the scores. (1 is fine, -1 is an outlier.)
outlier_scores = pd.DataFrame(lof.fit_predict(data))

# Append the score to the dataset.
data['Local Outlier Factor'] = outlier_scores

# We have a total of 36 outliers.
outliers = data.loc[data.loc[:, 'Local Outlier Factor'] == -1]

# Convert the state numbers back into names by first creating the dictionary of values.
state_names = {}
for index, value in enumerate(data_set):
    state_names[index] = value

# Now convert the state numbers back into names using the dictionary.
for value in list(set(outliers.STATE)):
    outliers.STATE.replace(value, state_names[value], inplace=True)

# Print out all the states with outliers.
print(outliers.STATE.value_counts())

# Remove the outliers from the dataset.
data_with_outliers = data.drop('Local Outlier Factor', axis=1)
data_no_outliers = data.loc[data.loc[:, 'Local Outlier Factor'] == 1].drop('Local Outlier Factor', axis=1)

''' Pairwise Correlations '''
#######################################################################################################################
# Calculate all the pairwise correlations
corr1 = data_no_outliers.corr()
corr2 = data_with_outliers.corr()

# Heat map of the different correlations.
sns.heatmap(corr1)
sns.heatmap(corr2)

# The statistical significance of each pairwise correlation.
correlations_no_outliers = {}
correlations_with_outliers = {}

columns = data_no_outliers.columns.tolist()
column2 = data_with_outliers.columns.tolist()

for a, b in itertools.combinations(columns, 2):
    correlations_no_outliers[a + '->' + b] = pearsonr(data_no_outliers.loc[:, a], data_no_outliers.loc[:, b])
for a, b in itertools.combinations(column2, 2):
    correlations_with_outliers[a + '->' + b] = pearsonr(data_with_outliers.loc[:, a], data_with_outliers.loc[:, b])

result_no_outliers = pd.DataFrame.from_dict(correlations_no_outliers, orient='index')
result_with_outliers = pd.DataFrame.from_dict(correlations_with_outliers, orient='index')

result_no_outliers.columns = ['Pearson Correlation', 'Statistical_Significance']
result_with_outliers.columns = ['Pearson Correlation', 'Statistical_Significance']

result_no_outliers = result_no_outliers.drop('Pearson Correlation', axis=1)
result_with_outliers = result_with_outliers.drop('Pearson Correlation', axis=1)

# Print the heatmaps.
sns.heatmap(result_no_outliers)
sns.heatmap(result_with_outliers)

''' Dimensionality Reduction '''
########################################################################################################################
# Initialize the PCA model.
pca = PCA()

# Perform the PCA on the non-outlier data.
pca.fit_transform(data_no_outliers)

# Plot the results.
plt.subplot(2, 1, 1)
plt.plot(pca.explained_variance_)
plt.xlabel('n_components')
plt.ylabel('explained variance')
plt.title('Explained Variance by Variable - No Outliers')

# Print out the explained variance.
print(pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance Ratio'], index=data_no_outliers.columns))

# Plot the PCA with the outlier data included.
plt.subplot(2, 1, 2)
pca.fit_transform(data_with_outliers)
plt.plot(pca.explained_variance_)
plt.xlabel('n_components')
plt.ylabel('explained variance')
plt.title('Explained Variance by Variable - With Outliers')
plt.tight_layout()

# Print out the explained variance
print(
    pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance Ratio'], index=data_with_outliers.columns))

''' Machine Learning. '''
########################################################################################################################
########################################################################################################################

''' Test/Train Split'''
# Initialize the k-fold.
group_kfold = GroupKFold()

# Create the group variable.
group = data_with_outliers.Year

# Create the X variable.
X = data_with_outliers

# Create the Test/Train datasets.
for train_index, test_index in group_kfold.split(X, y, group):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# We will create another X variable where the only the top 3 variance explaining variables are included.
X2 = data_with_outliers[['STATE', 'Year', 'Unemployment Rate']]

# Initialize the scores dataframe.
scores = pd.DataFrame(index=['Least Squares', 'Lasso', 'Elastic Net'],
                      columns=['Score with All Variables', 'Score with Top 3 Variables'])

''' Linear Regression. '''
########################################################################################################################

''' Least Squares Linear Regression '''
reglin = LinearRegression()

scores.loc['Least Squares', 'Score with All Variables'] = np.mean(cross_val_score(reglin, X, y, group, cv=GroupKFold()))
scores.loc['Least Squares', 'Score with Top 3 Variables'] = np.mean(
    cross_val_score(reglin, X2, y, group, cv=GroupKFold()))

''' Lasso Regression '''
reglas = Lasso()

scores.loc['Lasso', 'Score with All Variables'] = np.mean(cross_val_score(reglas, X, y, group, cv=GroupKFold()))
scores.loc['Lasso', 'Score with Top 3 Variables'] = np.mean(cross_val_score(reglas, X2, y, group, cv=GroupKFold()))

''' Elastic Net Regression'''
regenr = ElasticNet()

scores.loc['Elastic Net', 'Score with All Variables'] = np.mean(cross_val_score(regenr, X, y, group, cv=GroupKFold()))
scores.loc['Elastic Net', 'Score with Top 3 Variables'] = np.mean(
    cross_val_score(regenr, X2, y, group, cv=GroupKFold()))

''' Decision Trees '''
########################################################################################################################
scores_trees = pd.DataFrame(index=['Decision Tree Regressor', 'Random Forest Regressor'],
                            columns=['Score with All Variables', 'Score with Top 3 Variables'])

''' Decision Tree Regressor '''
clf = tree.DecisionTreeRegressor()

scores_trees.loc['Decision Tree Regressor', 'Score with All Variables'] = np.mean(
    cross_val_score(clf, X, y, group, cv=GroupKFold()))
scores_trees.loc['Decision Tree Regressor', 'Score with Top 3 Variables'] = np.mean(
    cross_val_score(clf, X2, y, group, cv=GroupKFold()))

''' Random Forest Regressor '''
clf2 = RandomForestRegressor()

scores_trees.loc['Random Forest Regressor', 'Score with All Variables'] = np.mean(
    cross_val_score(clf2, X, y, group, cv=GroupKFold()))
scores_trees.loc['Random Forest Regressor', 'Score with Top 3 Variables'] = np.mean(
    cross_val_score(clf2, X2, y, group, cv=GroupKFold()))

''' AdaBoost Regressor '''
clf3 = AdaBoostRegressor()

scores_trees.loc['AdaBoost Regressor', 'Score with All Variables'] = np.mean(cross_val_score(clf3, X, y,
                                                                                             group, cv=GroupKFold()))
scores_trees.loc['AdaBoost Regressor', 'Score with Top 3 Variables'] = np.mean(cross_val_score(clf3, X2, y,
                                                                                               group, cv=GroupKFold()))

''' Bagging Regressor '''
clf4 = BaggingRegressor()

scores_trees.loc['Bagging Regressor', 'Score with All Variables'] = np.mean(cross_val_score(clf4, X, y,
                                                                                            group, cv=GroupKFold()))
scores_trees.loc['Bagging Regressor', 'Score with Top 3 Variables'] = np.mean(cross_val_score(clf4, X2, y,
                                                                                              group, cv=GroupKFold()))
