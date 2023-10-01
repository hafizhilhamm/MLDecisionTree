import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

print("Age category : Young Adult (18-25), Middle Aged (25-45), Aged (45>)")
df = pd.read_csv('jobacceptance.csv') 

label_age = LabelEncoder()
label_edu = LabelEncoder()

df['Age'] = label_age.fit_transform(df['Age'])
df['Education'] = label_edu.fit_transform(df['Education'])

X = df[['Age', 'Education']]
y = df['Accepted?']

clf = GaussianNB()

clf.fit(X, y)

while True:
    user_age = input("Enter your age or 'exit' to quit: ")

    if user_age.lower() == 'exit':
        break

    user_edu = input("Enter Your Education: ")

    new_data = {
        'Age': [user_age],
        'Education': [user_edu]
    }

    new_df = pd.DataFrame(new_data)

    new_df['Age'] = label_age.transform(new_df['Age'])
    new_df['Education'] = label_edu.transform(new_df['Education'])

    predictions = clf.predict(new_df)

    for i in range(len(new_data['Age'])):
        print(f"Age: {new_data['Age'][i]}, Education: {new_data['Education'][i]}, Acceptance Prediction: {predictions[i]}")
