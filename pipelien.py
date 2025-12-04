from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline

steps01=[("scale", StandardScaler()),("linerar R", LinearRegression())]
steps02=[("encode", OneHotEncoder()), ("classify", RandomForestClassifier())]
pipe=make_pipeline(steps01)
pipe02=make_pipeline(steps02)
print(pipe02)
print(pipe)