import pandas as pd
import numpy as np
import re

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

# Slice by state and find the outliers.
list_of_states = list(set(final_poverty_dataset.STATE))

outliers = pd.DataFrame()
for index in range(len(list_of_states)):
    # Slice the data by the state.
    data = final_poverty_dataset[final_poverty_dataset.STATE == list_of_states[index]]

    # Iterate from Total to Percent columns (1:3)
    for i in [1, 2, 3]:
        IQR = (np.percentile(data[data.columns[i]], [75]) - np.percentile(data[data.columns[i]], [25]))[0]

        # Determine the upper and lower range.
        upper_range = np.percentile(data[data.columns[i]], [75])[0] + 1.5*IQR
        lower_range = np.percentile(data[data.columns[i]], [25])[0] - 1.5*IQR

        # Determine which state's values are outliers.
        if ((data[data.columns[i]] > upper_range) | (data[data.columns[i]] < lower_range)).any():
            outliers = outliers.append([[list_of_states[index], data.columns[i]]], ignore_index=True)

# Rename the outliers dataframe columns.
outliers.columns = ['State', 'Column']

# Return how many times a state has an outlier in one of it's columns.
outliers.State.value_counts()

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

# Export the two datasets to excel and combine them in excel.
final_poverty_dataset.to_excel('test.xlsx')
final_cardio_dataset.to_excel('test2.xlsx')

# Import the master dataset.
#final_data = pd.read_excel('master_dataset.xlsx')

