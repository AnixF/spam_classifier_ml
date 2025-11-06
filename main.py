import pandas as pd

sms = pd.read_csv("data/spam.csv")
sms = sms[['v1','v2']]
sms = ['label', 'text']

print(sms.head(5))