# logistica_analitica.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import linprog

# ========================
# 1. Dados simulados
# ========================

np.random.seed(42)

# CDs e Lojas
cds = ['CD_1', 'CD_2', 'CD_3']
lojas = ['Loja_A', 'Loja_B', 'Loja_C', 'Loja_D']

# Estoques disponíveis nos CDs
estoque_cds = np.random.randint(500, 1000, size=len(cds))

# Demandas das lojas (semanais)
demanda_lojas = np.random.randint(200, 600, size=len(lojas))

# Custos de transporte (matriz CDs x Lojas)
custos_transporte = np.random.randint(5, 20, size=(len(cds), len(lojas)))

# DataFrames para visualização
df_estoque = pd.DataFrame({'CD': cds, 'Estoque': estoque_cds})
df_demanda = pd.DataFrame({'Loja': lojas, 'Demanda': demanda_lojas})
df_custos = pd.DataFrame(custos_transporte, index=cds, columns=lojas)

# ========================
# 2. Análise Exploratória
# ========================

print("\n=== Estoque nos Centros de Distribuição ===")
print(df_estoque)

print("\n=== Demanda das Lojas ===")
print(df_demanda)

print("\n=== Custos de Transporte (por unidade) ===")
print(df_custos)

plt.figure(figsize=(8, 4))
sns.heatmap(df_custos, annot=True, cmap="YlGnBu")
plt.title("Matriz de Custo de Transporte")
plt.show()

# ========================
# 3. Previsão de Demanda (preditivo)
# ========================

# Simulando dados históricos para prever demanda
semanas = np.arange(1, 13).reshape(-1, 1)  # 12 semanas
demanda_historica = 50 + 10 * semanas.flatten() + np.random.normal(0, 10, size=12)  # tendência crescente

# Regressão Linear
model = LinearRegression()
model.fit(semanas, demanda_historica)
semana_13 = np.array([[13]])
previsao = model.predict(semana_13)[0]

print(f"\n🔮 Previsão de demanda para semana 13: {previsao:.0f} unidades")

plt.plot(semanas, demanda_historica, marker='o', label='Histórico')
plt.plot(semana_13, previsao, 'ro', label='Previsão')
plt.title("Previsão de Demanda")
plt.xlabel("Semana")
plt.ylabel("Unidades")
plt.legend()
plt.show()

# ========================
# 4. Otimização (prescritivo)
# ========================

# Flatten dos custos para o problema de programação linear
c = custos_transporte.flatten()

# Restrições de oferta (cada CD não pode enviar mais do que seu estoque)
A_ub = np.zeros((len(cds), len(cds) * len(lojas)))
for i in range(len(cds)):
    A_ub[i, i*len(lojas):(i+1)*len(lojas)] = 1
b_ub = estoque_cds

# Restrições de demanda (cada loja deve receber exatamente o que demanda)
A_eq = np.zeros((len(lojas), len(cds) * len(lojas)))
for j in range(len(lojas)):
    for i in range(len(cds)):
        A_eq[j, i*len(lojas) + j] = 1
b_eq = demanda_lojas

# Otimização com Programação Linear
res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')

if res.success:
    print("\n✅ Otimização realizada com sucesso!")
    print(f"Custo total de transporte: R${res.fun:.2f}")

    solucao = res.x.reshape(len(cds), len(lojas))
    df_resultado = pd.DataFrame(solucao, index=cds, columns=lojas).round()
    print("\n📦 Plano ótimo de transporte (unidades enviadas):")
    print(df_resultado)

    sns.heatmap(df_resultado, annot=True, cmap="Greens")
    plt.title("Plano de Transporte Otimizado")
    plt.show()
else:
    print("❌ A otimização falhou!")
