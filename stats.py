import json
import statistics

'''
compute statistics about pipeline results such as mean, variance, and standard deviation
'''

# Read data from JSON file
with open('results_22_12_23.json') as file:
    data = json.load(file)
    
r2_list = data['R2 Score List']
corr_list = data['Correlation List']

# Calculate statistics for r2_list
r2_mean = statistics.mean(r2_list)
r2_variance = statistics.variance(r2_list)
r2_std_dev = statistics.stdev(r2_list)

# Calculate statistics for corr_list
corr_mean = statistics.mean(corr_list)
corr_variance = statistics.variance(corr_list)
corr_std_dev = statistics.stdev(corr_list)

# Print the results for r2_list
print("R2 Score List:")
print("Mean:", r2_mean)
print("Variance:", r2_variance)
print("Standard Deviation:", r2_std_dev)

# Print the results for corr_list
print("Correlation List:")
print("Mean:", corr_mean)
print("Variance:", corr_variance)
print("Standard Deviation:", corr_std_dev)
