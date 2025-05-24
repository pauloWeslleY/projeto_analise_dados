# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# ========================
# 1. Análise Exploratória
# ========================

print("Informações do dataset:")
print(df.info())

print("\nResumo estatístico:")
print(df.describe())

# Correlação com o preço
correlation = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação")
plt.show()

# Distribuição dos preços
plt.figure(figsize=(8, 5))
sns.histplot(df['PRICE'], bins=30, kde=True)
plt.title("Distribuição dos Preços das Casas")
plt.xlabel("Preço ($1000)")
plt.ylabel("Frequência")
plt.show()

# Scatter plot entre RM (número de quartos) e PRICE
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='RM', y='PRICE')
plt.title("Preço vs Número de Quartos")
plt.xlabel("Número médio de quartos por residência")
plt.ylabel("Preço ($1000)")
plt.show()

# ========================
# 2. Modelagem Preditiva
# ========================

# Separar variáveis independentes e dependente
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Predição
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nErro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

# Comparar valores reais vs previstos
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Preço Real ($1000)")
plt.ylabel("Preço Previsto ($1000)")
plt.title("Comparação Preço Real vs Previsto")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # linha ideal
plt.show()
