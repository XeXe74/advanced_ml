import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


DATA = "../Datasets/breast cancer/wdbc.data"
names = "../Datasets/iris/iris.names"
encoder = LabelEncoder()
scaler = StandardScaler()

# The dataset has id, diagnosis and mean, standard error and worst for every column
features = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
columns = ["id", "diagnosis"] + [f"{name}_mean" for name in features] + [f"{name}_sem" for name in features] + [f"{name}_worst" for name in features]
wdbc = pd.read_csv(DATA, names=columns)

print(wdbc.head())
print("\n")

wdbc.info()
print("\n")

print(wdbc.describe())
print("\n")

print(f"Duplicated data: {wdbc.duplicated().sum()}")

nulls = wdbc.isnull().sum().sort_values(ascending=False)
print(f"Null data:\n{nulls}")

df_numeric = wdbc.select_dtypes(include="number")

# Pearson Correlation
correlation_matrix = df_numeric.corr(method="pearson")
print(correlation_matrix)

plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosis', data=wdbc, palette='pastel')
plt.title("Diagnosis Distribution")
plt.show()

X = wdbc.drop(columns=["id", "diagnosis"]) # Features
y = encoder.fit_transform(wdbc["diagnosis"]) # Label

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dictionary for trying all the classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter= 1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVC": SVC(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

# Loop for testing the different models
for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Model: {name}")
    print(f"Confussion matrix:\n {confusion_matrix(y_test, y_pred)}\n")
    print(f"Classification report:\n {classification_report(y_test, y_pred)}\n")

# Some types of layers for trying them out in the different Perceptrons
layers = [
    (10,),
    (100,),
    (50, 50),
    (100, 50, 25)
]

for layer in layers:

    mlp = MLPClassifier(hidden_layer_sizes=layer, max_iter=1000)

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    print(f"Layers: {layer}")
    print(f"Confussion matrix:\n {confusion_matrix(y_test, y_pred)}\n")
    print(f"Classification report:\n {classification_report(y_test, y_pred)}\n")