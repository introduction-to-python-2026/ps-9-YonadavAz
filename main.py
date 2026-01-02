import pandas
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# get data
data = pandas.read_csv("parkinsons.csv")

# clean data
data = data.dropna()
# chose features
features = ["MDVP:Fo(Hz)", "RPDE"]
label_feature = ["status"]
# get features
X = data[features]
y = data[label_feature]
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scale data
# scaler = MinMaxScaler()
# X_scaled_train = scaler.fit_transform(X_train)

# y_train should not be scaled for classification; convert to 1D array
y_train_flat = y_train.values.ravel()

# build the model
# model = KNeighborsClassifier(n_neighbors=12)
# model.fit(X_scaled_train, y_train_flat)
# for d in range(1, 8):
#     model = DecisionTreeClassifier(max_depth=d, random_state=42)
#     model.fit(X_train, y_train_flat)
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(d, acc)
model = DecisionTreeClassifier(max_depth=7, random_state=42)
model.fit(X_train, y_train_flat)
# Scale X_test using the same scaler fitted on X_train
# X_scaled_test = scaler.transform(X_test)
# test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


