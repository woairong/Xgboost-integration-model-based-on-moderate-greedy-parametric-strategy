import pandas as pd

df1 = pd.read_csv('G:\\ml360\\train\\traindata1.txt', sep='\t')
df2 = pd.read_csv('G:\\ml360\\train\\traindata2.txt', sep='\t')
df3 = pd.read_csv('G:\\ml360\\train\\traindata3.txt', sep='\t')
df4 = pd.read_csv('G:\\ml360\\train\\traindata4.txt', sep='\t')
df5 = pd.read_csv('G:\\ml360\\train\\traindata5.txt', sep='\t')
df1.to_csv('G:\\ml360\\train\\1104.csv', index = False, sep = ',')
df20 = df2.ix[0:13463,:]
df1.to_csv('G:\\ml360\\train\\1104.csv', index = False, sep = ',')
df20.to_csv('G:\\ml360\\train\\1104.csv', mode = 'a', index = False, header = False, sep = ',')
df1104 = pd.read_csv('G:\\ml360\\train\\1104.csv')

