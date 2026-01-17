from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

data=pd.read_csv("dataset.csv")
x=data.drop(['target_str',"target","cp_type","sex_str"],axis=1).astype(float)
y=data["target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

model1=RandomForestClassifier()
model2=SVC()
model3=LogisticRegression()
vote=VotingClassifier(
    estimators=[("rfc",model1),("svc", model2),("lr",model3)],
    voting="hard"
                )
vote.fit(x_train,y_train)
y_pred=vote.predict(x_test)
print(accuracy_score(y_pred,y_test))