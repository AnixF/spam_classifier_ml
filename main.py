import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


sms = pd.read_csv("data/spam.csv")
sms = sms[['v1','v2']]
sms.columns = ['label', 'text']

x = sms['text']
y = sms['label']

vectorizer = CountVectorizer()
x_vec = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.1f}%")

def check_sms(sms):
    new_sms = [sms]
    new_vec = vectorizer.transform(new_sms)
    result = model.predict(new_vec)
    return result[0]

print(check_sms("Congrats! U won 100000000 dollars, take it FREE from this website: http://hackyourphone.com"))

user_sms = input("Enter sms: ")
print(check_sms(user_sms))