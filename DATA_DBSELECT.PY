import pandas as pd

#Load the CSV file
df = pd.read_csv('D:/University/Year 3/CS3072-CS3605 FYP/Perseus-FYP/Datasets/CIC-DDoS2019/Syn.csv')  # uploading the CIC DDOS 2019

#removing the white spaces 
df.columns = df.columns.str.strip()


#Filter and sample
benign_sample = df[df['Label'] == 'BENIGN'].sample(n=5000, random_state=1)  # Sample 5000 BENIGN rows
syn_sample = df[df['Label'] == 'Syn'].sample(n=5000, random_state=1)  # Sample 5000 SYN rows

# Concatenate the DataFrames
final_sample = pd.concat([benign_sample, syn_sample])

# Saveing the new CSV file
# final_sample.to_csv('D:/University/Year 3/CS3072-CS3605 FYP/Perseus-FYP/Datasets/CIC-DDoS2019/Balanced-SYN-V2.csv', index=False)  