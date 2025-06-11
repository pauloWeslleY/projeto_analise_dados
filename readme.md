# 🚚 Otimização de Logística e Cadeia de Suprimentos com Python

Este projeto é uma aplicação analítica desenvolvida em Python para simular e otimizar a cadeia de suprimentos entre centros de distribuição (CDs) e lojas. Ele aborda desde a **análise exploratória de dados (EDA)** até a **modelagem preditiva**, **prescritiva** e **recomendações operacionais** com foco em decisões logísticas inteligentes.

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

- 📈 **Visualizações Informativas**:

  - Gráfico de barras do **estoque restante** em cada CD após o transporte.
  - Comparação visual entre **demanda histórica** e **previsão futura**.

- 💡 **Recomendações Operacionais**:
  - Sugestões práticas com base nos resultados da otimização, incluindo:
    - CDs com estoque excedente.
    - Tendências de crescimento ou queda da demanda.
    - Avaliação do custo de transporte e alternativas logísticas.

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
- Gráficos de tendência de demanda e previsão para semanas futuras.
- Matriz de transporte otimizada com envio ideal entre CDs e lojas.
- Gráfico de estoque restante por CD.
- Custo total de transporte mínimo calculado automaticamente.
- Recomendações automáticas com base nos resultados da otimização e previsão.

---

## 🧩 Extensões Possíveis

Você pode expandir este projeto para incluir:

- Múltiplos produtos.
- Restrições de capacidade de veículos.
- Priorização por tempo de entrega.
- Visualização interativa com **Streamlit** ou **Dash**.
- Conexão com bases de dados reais (ex: via SQL ou APIs).
- Geração de relatórios automáticos (ex: PDF, PowerPoint, dashboards).

---

## 🧠 Sobre

Este projeto tem fins didáticos e de prototipagem analítica, sendo ideal para profissionais e estudantes que queiram entender como aplicar **Machine Learning**, **Análise de Dados** e **Otimização** no contexto de **logística e supply chain**.

---
