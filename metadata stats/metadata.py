import pandas as pd

# Load the Excel file
df = pd.read_excel('metadata2023_08.xlsx')

# Compute statistics on age
age_stats = df['age_aha'].describe()

# Compute statistics on sex
sex_stats = df['gender'].value_counts()

# Compute statistics on sex
hemi_stats = df['hemi'].value_counts()

# Print the statistics
print("Age Statistics:")
print(age_stats)
print("\nSex Statistics:")
print(sex_stats)
print("\nHemi Statistics:")
print(hemi_stats)

# Compute statistics on age for hemi group == 2
age_stats_hemi2 = df[df['hemi'] == 2]['age_aha'].describe()

# Compute statistics on age for hemi group == 1
age_stats_hemi1 = df[df['hemi'] == 1]['age_aha'].describe()

print("\nAge Statistics for Hemi Group == 2:")
print(age_stats_hemi2)
print("\nAge Statistics for Hemi Group == 1:")
print(age_stats_hemi1)