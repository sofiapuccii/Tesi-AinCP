import os
import pandas as pd

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('file.csv')
print(df)