import pandas as pd
import utils
from sklearn import linear_model,preprocessing

train = pd.read_csv("train.csv")
utils.clean_data(train)

target = train["Survived"].values
features_name = ["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]
features = train[features_name].values

classifire = linear_model.LogisticRegression()

classifire_ = classifire.fit(features,target)
print(classifire_.score(features,target))

poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

classifire_ = classifire.fit(poly_features,target)
print(classifire_.score(poly_features,target))