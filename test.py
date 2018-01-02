# Calculate the number of outliers in the poverty rate and mortality rate columns.
for i in [3, 5]:
    outliers = pd.DataFrame()
    # Determine the IQR.
    IQR = (np.percentile(data[data.columns[i]], [75]) - np.percentile(data[data.columns[i]], [25]))[0]

    # Determine the upper and lower range.
    upper_range = np.percentile(data[data.columns[i]], [75])[0] + 1.5*IQR
    lower_range = np.percentile(data[data.columns[i]], [25])[0] - 1.5*IQR

    # Determine which state's values are outliers.
    outliers = outliers.append(data[data[data.columns[i]] > upper_range])
    outliers = outliers.append(data[data[data.columns[i]] < lower_range])

    # Add what kind of outlier column.
    if outliers.empty:
        continue
    else:
        if i == 3:
            outliers['Outlier'] = 'Percent'
        else:
            outliers['Outlier'] = 'Mortality'
    outliers_total_state = outliers_total.append(outliers)


print(outliers_total_state)