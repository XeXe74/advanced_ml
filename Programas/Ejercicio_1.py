import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

DATA = "../Datasets/iris/iris.data"
names = "../Datasets/iris/iris.names"
encoder = LabelEncoder()

iris = pd.read_csv(DATA)

"""with open(names, 'r') as f:
    print(f.read())
    print("\n")"""

print(iris.head())
print("\n")

iris.info()
print("\n")

print(iris.describe())
print("\n")

print(f"Número de valores duplicados: {iris.duplicated().sum()}")

nulos = iris.isnull().sum().sort_values(ascending=False)
print(f"Número de valores nulos por columna:\n{nulos}")

df_numeric = iris.select_dtypes(include="number")

# Correlación de Pearson entre todas las columnas
matriz_correlacion = df_numeric.corr(method="pearson")
print(matriz_correlacion)

y = encoder.fit_transform(iris["Iris-setosa"])

print(y)


