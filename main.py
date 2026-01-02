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
X = data[features]
y = data["status"]
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scale data
scaler = MinMaxScaler()
X_scaled_train = scaler.fit_transform(X_train)

# y_train should not be scaled for classification; convert to 1D array
y_train_flat = y_train.values.ravel()
X_train_flat = X_train.values.ravel()
# build the model
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_scaled_train, y_train_flat)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# for k in range(1, 21):
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"k={k}  accuracy={acc:.3f}")
# model = DecisionTreeClassifier(max_depth=7, random_state=42)
# model.fit(X_train, y_train_flat)
model = KNeighborsClassifier(n_neighbors=12)
model.fit(X_train_scaled, y_train)
# test the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


