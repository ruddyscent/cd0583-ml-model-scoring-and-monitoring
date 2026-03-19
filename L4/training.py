import pandas as pd
from sklearn.linear_model import LogisticRegression

trainingdata = pd.read_csv('samplefileingested.csv')

X = trainingdata.loc[:, ['col1', 'col2']].values.reshape(-1, 2)
y = trainingdata['col3'].values.reshape(-1, 1).ravel()

model = LogisticRegression(
    solver='liblinear',
    random_state=0
).fit(X, y)