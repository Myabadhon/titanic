import pandas as pd
import utils
from sklearn import tree,model_selection

train = pd.read_csv("train.csv")
utils.clean_data(train)

target = train["Survived"].values
features_name = ["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]
features = train[features_name].values

decesion_tree = tree.DecisionTreeClassifier(random_state=1)
decesion_tree_ = decesion_tree.fit(features,target)

print(decesion_tree_.score(features,target))

scores = model_selection.cross_val_score(decesion_tree, features, target, scoring="accuracy", cv = 50)
print(scores)
print(scores.mean())

generalized_tree = tree.DecisionTreeClassifier(
    random_state = 1,
    max_depth = 7,
    min_samples_split = 2,
)
generalized_tree_ = generalized_tree.fit(features,target)

print(generalized_tree_.score(features,target))

scores = model_selection.cross_val_score(generalized_tree, features, target, scoring="accuracy", cv = 50)
print(scores)
print(scores.mean())

tree.export_graphviz(generalized_tree, feature_names=features_name, out_file="tree.dot")