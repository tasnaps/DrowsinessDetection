import pandas as pd
import matplotlib.pyplot as plt
# Load the CSV file into a DataFrame
df = pd.read_csv("C:\\Users\\tapio\\Desktop\\data\\merged-csv.csv")

# Show the first 5 rows of the DataFrame
print(df.head())

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
kss_counts = df['KSS'].value_counts()
print(kss_counts)
plt.figure(figsize=(10, 6))
plt.hist(df['Model Score'], bins=2, edgecolor='k')
plt.xlabel('Awake vs Drowsy')
plt.ylabel('Frequency')
plt.title('Histogram of Model Score')
plt.show()