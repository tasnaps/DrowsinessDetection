import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV file into a DataFrame, with error handling for bad lines
df = pd.read_csv("C:\\Users\\tapio\\Desktop\\data\\merged-csv.csv", on_bad_lines='skip')

# Show the first 5 rows of the DataFrame
print(df.head())

# Convert 'Date' and 'Time' to datetime format explicitly
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S")

# Drop rows where Date or Time conversion failed
df = df.dropna(subset=['Date', 'Time'])

# Handle missing values if any
df = df.dropna()

# Summary statistics
print(df.describe())

# Prepare a histogram for 'KSS'
plt.figure(figsize=(10, 6))
plt.hist(df['KSS'], bins=9, edgecolor='k')
plt.xlabel('KSS')
plt.ylabel('Frequency')
plt.title('Histogram of KSS')
plt.show()

# Calculate the average of awakeness count
awakeness_avg = df['Awakeness Count'].mean()
print(f'Average Awakeness Count: {awakeness_avg}')

# Calculate the average of drowsy count
drowsy_avg = df['Drowsiness Count'].mean()
print(f'Average Drowsy Count: {drowsy_avg}')

# Find out the maximum mean confidence
max_mean_confidence = df['Mean Confidence'].max()
print(f'Maximum Mean Confidence: {max_mean_confidence}')

# Find out the maximum std confidence
max_std_confidence = df['Standard Deviation Confidence'].max()
print(f'Maximum Std Confidence: {max_std_confidence}')

# Prepare a column 'Model Score' that represents model's output in KSS scale.
# If 'Drowsiness Count' > 'Awakeness Count', it's 'Drowsy', else 'Awake'
df['Model Score'] = df.apply(lambda row: 9 if row['Drowsiness Count'] > row['Awakeness Count'] else 1, axis=1)

# Exclude rows where KSS is in between 4-6 as we can't clearly mark it as 'Awake' or 'Drowsy'.
df = df[(df['KSS'] <= 3) | (df['KSS'] >= 7)]

# Replace KSS scores in the range 1-3 with 'awake' and 7-9 with 'drowsy'
df['KSS'] = df['KSS'].apply(lambda x: 1 if x <= 3 else 9)

# Calculate correlation
correlation = df['KSS'].corr(df['Model Score'])
print(f'Correlation between KSS and Model Score: {correlation}')

# Visualize the distribution of KSS and Model Score
plt.figure(figsize=(10, 6))
plt.hist(df['Model Score'], bins=2, edgecolor='k')
plt.xlabel('Awake vs Drowsy')
plt.ylabel('Frequency')
plt.title('Histogram of Model Score')
plt.show()

# Visualize the correlation matrix
df_numeric = df._get_numeric_data()
with np.errstate(divide='ignore', invalid='ignore'):
    corr_matrix = df_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Additional EDA: pairplot to visualize relationships
sns.pairplot(df[['KSS', 'Model Score', 'Awakeness Count', 'Drowsiness Count', 'Mean Confidence',
                 'Standard Deviation Confidence']])
plt.show()

# Analyzing based on time periods
df['Hour'] = pd.to_datetime(df['Time'], format="%H:%M:%S").dt.hour

# Create masks for different time periods
mask_morning = (df['Hour'] >= 6) & (df['Hour'] < 12)
mask_afternoon = (df['Hour'] >= 12) & (df['Hour'] < 18)

# Subset the data
df_morning = df.loc[mask_morning]

# Drop rows with missing values in sub-datasets
df_morning = df_morning.dropna()

# Add print statements to check the contents of df_morning and df_afternoon
print(f"Mornings data (first 5 rows): \n{df_morning.head()}")

# Before calculating correlation, check if there is no variance in Model Score
if df_morning['Model Score'].std() != 0:
    correlation_morning = df_morning['KSS'].corr(df_morning['Model Score'])
    print(f'Correlation between KSS and Model Score (Morning): {correlation_morning}')
else:
    print("Cannot compute correlation in Morning data because all Model Scores are the same.")

df_afternoon = df.loc[mask_afternoon]

# Drop rows with missing values in sub-datasets
df_afternoon = df_afternoon.dropna()

# Add print statements to check the contents of df_morning and df_afternoon
print(f"Afternoon data (first 5 rows): \n{df_afternoon.head()}")

# Same check and calculation for df_afternoon
if df_afternoon['Model Score'].std() != 0:
    correlation_afternoon = df_afternoon['KSS'].corr(df_afternoon['Model Score'])
    print(f'Correlation between KSS and Model Score (Afternoon): {correlation_afternoon}')
else:
    print("Cannot compute correlation in Afternoon data because all Model Scores are the same.")
