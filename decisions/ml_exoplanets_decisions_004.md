# Relatório de Decisão — Algoritmos de Classificação Selecionados

**Projeto:** Classificação de Exoplanetas (KOI — Kepler Objects of Interest)
**Data:** 01/08/2026

## Documento de Decisão #004

## 1. Contexto

Definido o problema como uma classificação binária (`CONFIRMED` vs `FALSE POSITIVE`), com 12 features numéricas contínuas já padronizadas (`StandardScaler`) e um conjunto de dados conhecidos de 9.562 registros (7.586 para treino, 1.976 para teste, com divisão estratificada), foi necessário selecionar quais algoritmos de classificação seriam treinados e comparados.

Importante destacar a diferença conceitual: **classificação é o tipo de tarefa** (prever uma categoria), não um algoritmo específico. Diversos algoritmos resolvem esse mesmo tipo de tarefa, cada um com uma lógica matemática distinta, e por isso apresentam desempenhos diferentes conforme a natureza dos dados.

## 2. Algoritmos selecionados

### 2.1 Regressão Logística
**Papel:** baseline simples e interpretável.

Ajusta uma equação linear às features e aplica uma função sigmoide para estimar a probabilidade de cada classe. Serve como ponto de referência: se performar bem, indica que as classes são razoavelmente separáveis de forma linear; se performar mal em comparação aos demais, reforça a existência de relações não lineares nos dados. Seus coeficientes também são diretamente interpretáveis, o que agrega valor científico à análise.

### 2.2 Random Forest
**Papel:** principal candidato para o resultado final.

Constrói múltiplas árvores de decisão e combina seus resultados por votação. Lida bem com relações não lineares entre as features — comuns em dados astrofísicos, como a relação entre profundidade de trânsito e raio do planeta — e é robusto a outliers residuais. Também fornece a importância relativa de cada feature (`feature_importances_`), permitindo identificar quais variáveis mais influenciam a classificação.

### 2.3 SVM (kernel RBF)
**Papel:** alternativa não linear complementar ao Random Forest.

Busca a melhor fronteira de separação entre as classes no espaço das features. Como as features já estão padronizadas — pré-requisito para bom funcionamento do SVM — sua aplicação é viável. Pode capturar fronteiras de decisão complexas, ainda que exija mais ajuste de hiperparâmetros (`C`, `gamma`) para ser competitivo frente ao Random Forest.

### 2.4 Gradient Boosting (opcional, extensão além do currículo do curso de referência)
**Papel:** modelo adicional alinhado à literatura científica sobre o problema.

Amplamente utilizado em trabalhos e competições sobre classificação de KOIs, costuma superar o Random Forest em problemas tabulares semelhantes a este. Sua inclusão reforça a análise, embora não seja obrigatória para a comparação inicial.

## 3. Algoritmos considerados e descartados

| Algoritmo | Motivo da não seleção |
|---|---|
| Naive Bayes | Assume independência entre as features, premissa pouco realista aqui — várias features possuem correlação física esperada (ex.: raio estelar e gravidade superficial da estrela) |
| KNN | Sensível a desbalanceamento de classes e à alta dimensionalidade (12 features); custo computacional elevado na predição |
| Redes Neurais (MLP) | Tipicamente exige volumes de dados maiores para superar modelos como Random Forest/Gradient Boosting em dados tabulares; considerado possível trabalho futuro |

## 4. Metodologia de comparação

Os algoritmos selecionados (Regressão Logística, Random Forest e SVM, com Gradient Boosting como extensão) serão treinados sobre o mesmo conjunto de treino e avaliados sobre o mesmo conjunto de teste, utilizando as métricas: acurácia, precisão, recall, F1-score e matriz de confusão. O modelo com melhor desempenho nessas métricas será utilizado como resultado principal para a aplicação sobre os dados `CANDIDATE`; os demais modelos também serão aplicados aos candidatos, como forma de validação cruzada qualitativa por concordância entre modelos.