import pandas
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


data = pandas.read_csv("/content/parkinsons.csv")
data = data.dropna()
from matplotlib import pyplot
features = ["MDVP:Fo(Hz)", "RPDE"]
label_feature = ["status"]
X = data[features]
y = data[label_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_scaled_train = scaler.fit_transform(X_train)

# y_train should not be scaled for classification; convert to 1D array
y_train_flat = y_train.values.ravel()

model = KNeighborsClassifier(n_neighbors=12)
model.fit(X_scaled_train, y_train_flat)


# Scale X_test using the same scaler fitted on X_train
X_scaled_test = scaler.transform(X_test)

y_pred = model.predict(X_scaled_test)
accuracy = accuracy_score(y_test, y_pred)
