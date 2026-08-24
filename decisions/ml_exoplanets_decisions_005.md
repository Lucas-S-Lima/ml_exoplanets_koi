# ml_exoplanets_decisions_005

# TCC: Classificação de Exoplanetas Utilizando Machine Learning Aplicado aos Dados da Missão Kepler

## Documento de Decisão #005

**Assunto:** Definição do processo de tuning de hiperparâmetros do Random Forest (GridSearchCV) e do critério de avaliação, na ausência de um critério de desempate fixo entre modelos

**Data:** 24/08/2026

---

## Contexto

Após a análise da curva de erro OOB (*out-of-bag*) do Random Forest sobre o problema binário CONFIRMED vs. FALSE POSITIVE (base Cumulative KOI), observou-se que o erro não estabiliza de forma tão nítida quanto em outros datasets testados anteriormente (ex.: base de crédito). A curva apresentou:

- Queda acentuada do erro OOB entre 10 e 75 árvores;
- Queda mais suave e gradual entre 75 e 300 árvores, com o menor erro observado em `n_estimators = 300` (~0,075);
- Oscilação sem tendência clara de melhora entre 300 e 500 árvores (ruído estatístico do processo de bootstrap).

Além disso, foi definido que **não haverá um critério de desempate fixo** entre os quatro modelos candidatos (Regressão Logística, Random Forest, SVM e Gradient Boosting). A avaliação final será feita de forma geral, comparando as métricas de todos os modelos após os testes, em vez de priorizar uma métrica/classe específica (como o recall da classe CONFIRMED, cogitado anteriormente).

## Decisão

1. **Faixa de `n_estimators` no GridSearchCV restrita a `[150, 200, 300]`**, com base no resultado da curva OOB, evitando reavaliar valores já descartados (10, 25, 50, etc.).
2. **Múltiplas métricas de scoring no GridSearchCV**: `accuracy`, `precision_macro`, `recall_macro` e `f1_macro`, permitindo inspecionar o comportamento do modelo sob diferentes ângulos via `cv_results_`.
3. **Métrica de refit: `f1_macro`**. Por não haver critério de desempate fixo, optou-se por uma métrica que trata as classes CONFIRMED e FALSE POSITIVE com peso igual (média não ponderada dos F1-scores de cada classe), evitando que a classe majoritária domine a seleção do "melhor" modelo apenas por ter mais amostras.
4. **`criterion='entropy'`** mantido fixo (fora da grade de busca), com base em decisão anterior de que a diferença de desempenho entre `gini` e `entropy` costuma ser marginal.
5. Demais hiperparâmetros incluídos na busca: `max_depth`, `min_samples_leaf`, `max_features` e `class_weight`.

## Justificativa

- O erro OOB é adequado para decidir isoladamente o hiperparâmetro `n_estimators` de forma rápida (sem custo de validação cruzada completa), mas não substitui a análise final de métricas por classe — por isso seu uso foi limitado a essa etapa preliminar de definição da faixa de busca.
- A escolha de `f1_macro` como métrica de refit é consistente com a decisão de não priorizar uma classe específica: garante que o modelo selecionado pelo GridSearchCV performe bem em ambas as classes, e não apenas na mais frequente.
- Restringir a faixa de `n_estimators` reduz o espaço de busca do GridSearchCV (evitando testar novamente valores já avaliados pela curva OOB), tornando o processo computacionalmente mais viável.

## Consequências

- A seleção final entre os quatro modelos (Regressão Logística, Random Forest, SVM, Gradient Boosting) será feita por comparação qualitativa e quantitativa das métricas (acurácia, precisão, recall e F1 por classe), e não por um critério único de desempate.
- O `best_estimator_` retornado pelo GridSearchCV do Random Forest será selecionado com base em `f1_macro`, mas todas as demais métricas (`accuracy`, `precision_macro`, `recall_macro`) permanecerão disponíveis em `cv_results_` para inspeção manual antes da comparação final entre modelos.