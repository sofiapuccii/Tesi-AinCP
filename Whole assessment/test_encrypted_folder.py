import os
import pandas as pd

# Cambio la directory di esecuzione in quella dove si trova questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('file.csv')
print(df)

"""
import gnupg
import csv
import io

# Initialize GnuPG
gpg = gnupg.GPG()

# Decrypt the file
with open('yourfile.csv.gpg', 'rb') as f:
    decrypted_data = gpg.decrypt_file(f, passphrase='your_passphrase')

# Check if decryption was successful
if not decrypted_data.ok:
    raise ValueError("Decryption failed")

# Read the decrypted data as CSV
csv_reader = csv.reader(io.StringIO(str(decrypted_data)))
for row in csv_reader:
    print(row)
"""