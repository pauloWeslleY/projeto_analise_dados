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


# ========================
# 5. Visualizações Claras e Informativas
# ========================

if res.success:
    # Estoque restante após transporte
    estoque_utilizado = df_resultado.sum(axis=1).values
    estoque_restante = estoque_cds - estoque_utilizado

    df_estoque_restante = pd.DataFrame({
        'CD': cds,
        'Estoque Inicial': estoque_cds,
        'Utilizado': estoque_utilizado.astype(int),
        'Restante': estoque_restante.astype(int)
    })

    print("\n📊 Estoque restante após transporte:")
    print(df_estoque_restante)

    plt.figure(figsize=(8, 4))
    sns.barplot(x='CD', y='Restante', data=df_estoque_restante, palette="Blues_d")
    plt.title("Estoque Restante por CD após Transporte")
    plt.ylabel("Unidades")
    plt.show()

    # Comparação de demanda prevista vs. real
    plt.figure(figsize=(8, 4))
    plt.plot(semanas.flatten(), demanda_historica, marker='o', label='Demanda Histórica')
    plt.axhline(y=previsao, color='red', linestyle='--', label=f'Previsão Semana 13 ({previsao:.0f})')
    plt.title("Demanda Histórica vs. Previsão")
    plt.xlabel("Semana")
    plt.ylabel("Unidades")
    plt.legend()
    plt.show()

# ========================
# 6. Recomendações Práticas Baseadas nos Resultados
# ========================

if res.success:
    print("\n💡 Recomendações Operacionais:")

    # CDs com sobra de estoque
    for idx, cd in enumerate(cds):
        restante = estoque_restante[idx]
        if restante > 0:
            print(f"• {cd}: Ainda possui {restante} unidades em estoque. Pode ser utilizado em futuras demandas ou realocado.")

    # Análise da previsão
    if previsao > demanda_historica[-1]:
        print("• A previsão de demanda para a próxima semana indica uma tendência de crescimento. Considere reforçar os estoques nas lojas mais demandadas.")
    else:
        print("• A demanda prevista está estável ou em queda. Reavalie a necessidade de envio excessivo de produtos.")

    # Custo alto
    if res.fun > 0.9 * (custos_transporte.max() * demanda_lojas.sum()):
        print("• O custo total de transporte está relativamente alto. Avalie a possibilidade de revisar contratos logísticos ou reorganizar rotas.")

    print("• Considere utilizar esta análise regularmente com dados reais atualizados para melhor tomada de decisão logística.")
