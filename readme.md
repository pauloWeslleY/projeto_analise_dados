# 🚚 Otimização de Logística e Cadeia de Suprimentos com Python

Este projeto é uma aplicação analítica desenvolvida em Python para simular e otimizar a cadeia de suprimentos entre centros de distribuição (CDs) e lojas. Ele aborda desde a **análise exploratória de dados (EDA)** até a **modelagem preditiva** e **prescritiva** com foco em decisões logísticas.

---

## 🔍 Funcionalidades

- 📊 **Análise Exploratória de Dados**:

  - Visualização de estoques, demandas e custos de transporte.
  - Geração de mapas de calor para entender os custos entre CDs e lojas.

- 🔮 **Modelagem Preditiva**:

  - Previsão de demanda semanal usando Regressão Linear com `scikit-learn`.

- 🧠 **Modelagem Prescritiva**:
  - Otimização do plano de transporte utilizando Programação Linear (`scipy.optimize.linprog`) para minimizar os custos logísticos.
  - Restrições de oferta e demanda respeitadas.

---

## 🗂️ Estrutura do Projeto

```
logistica_analitica.py     # Código principal da aplicação
README.md                  # Este arquivo
```

---

## 📦 Bibliotecas Utilizadas

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`

Instale tudo com:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## ▶️ Como Executar

1. Clone este repositório ou copie o arquivo `logistica_analitica.py`.

2. Instale as dependências (veja acima).

3. Execute o script:

```bash
python logistica_analitica.py
```

---

## 📈 Exemplos de Saída

- Tabelas com estoques, demandas e custos de transporte.
- Gráficos de tendência de demanda.
- Matriz de transporte otimizada com envio ideal entre CDs e lojas.
- Custo total de transporte mínimo calculado automaticamente.

---

## 🧩 Extensões Possíveis

Você pode expandir este projeto para incluir:

- Múltiplos produtos.
- Restrições de capacidade de veículos.
- Priorização por tempo de entrega.
- Visualização interativa com **Streamlit** ou **Dash**.

---

## 🧠 Sobre

Este projeto tem fins didáticos e de prototipagem analítica, sendo ideal para profissionais e estudantes que queiram entender como aplicar **Machine Learning** e **Otimização** no contexto de **logística e supply chain**.

---
