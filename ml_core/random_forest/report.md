Relatório de Resultados — Random Forest
1. Configuração do modelo

O modelo utilizado foi o Random Forest Classifier, com os hiperparâmetros selecionados pelo processo de Grid Search:

| Hiperparâmetro     | Valor                  |
| ------------------ | ---------------------- |
| `n_estimators`     | **200**                |
| `criterion`        | **entropy**            |
| `class_weight`     | **balanced_subsample** |
| `max_depth`        | **30**                 |
| `min_samples_leaf` | **1**                  |
| `max_features`     | **sqrt**               |
| `random_state`     | **42**                 |

A seleção dos hiperparâmetros foi realizada por meio de validação cruzada estratificada com 5 folds, avaliando 216 combinações diferentes.

O critério utilizado para seleção do melhor modelo foi o F1 macro, por atribuir o mesmo peso às duas classes e, portanto, ser mais adequado que a accuracy isoladamente em um conjunto de dados com distribuição desigual entre as classes.

2. Resultado da validação cruzada

O melhor conjunto de hiperparâmetros apresentou:

| Métrica         |  Resultado |
| --------------- | ---------: |
| Accuracy        |     0,9246 |
| Precision macro |     0,9174 |
| Recall macro    |     0,9202 |
| **F1 macro**    | **0,9187** |

O melhor modelo foi obtido com:

| Hiperparâmetro     | Valor                |
| ------------------ | -------------------- |
| `class_weight`     | `balanced_subsample` |
| `max_depth`        | `30`                 |
| `max_features`     | `sqrt`               |
| `min_samples_leaf` | `1`                  |
| `n_estimators`     | `200`                |

É importante destacar que n_estimators=200 apresentou desempenho praticamente equivalente a 300 árvores.

| Número de árvores | Melhor F1 macro |
| ----------------: | --------------: |
|               150 |        0,916790 |
|           **200** |    **0,918670** |
|               300 |        0,918622 |


A diferença entre 200 e 300 árvores foi de aproximadamente 0,00005, sendo extremamente pequena no contexto deste experimento.

Portanto, o aumento de 200 para 300 árvores não proporcionou ganho relevante de desempenho, embora aumente o custo computacional.

3. Avaliação no conjunto de teste

Após a seleção dos hiperparâmetros, o modelo final foi treinado utilizando o conjunto de treinamento e posteriormente avaliado no conjunto de teste.

O conjunto de teste contém 1.897 amostras, distribuídas da seguinte forma:

| Classe    |  Amostras |
| --------- | --------: |
| Classe 0  |       687 |
| Classe 1  |     1.210 |
| **Total** | **1.897** |


Os resultados obtidos foram:

| Classe           | Precision |   Recall | F1-score |   Support |
| ---------------- | --------: | -------: | -------: | --------: |
| **0**            |      0,90 |     0,89 |     0,90 |       687 |
| **1**            |      0,94 |     0,94 |     0,94 |     1.210 |
| **Macro avg**    |  **0,92** | **0,92** | **0,92** |     1.897 |
| **Weighted avg** |  **0,92** | **0,92** | **0,92** |     1.897 |
| **Accuracy**     |           |          | **0,92** | **1.897** |

A accuracy final foi de 0,9246, correspondente a aproximadamente 92,46% de classificações corretas.

4. Matriz de confusão

A matriz de confusão permite analisar diretamente os acertos e erros cometidos pelo modelo em cada classe.

Figura 1 — Matriz de confusão do modelo Random Forest no conjunto de teste.

A matriz obtida foi:

|             | Predito: 0 | Predito: 1 |
| ----------- | ---------: | ---------: |
| **Real: 0** |    **613** |     **74** |
| **Real: 1** |     **69** |  **1.141** |


Os valores podem ser interpretados da seguinte maneira:

613 amostras da classe 0 foram corretamente classificadas como classe 0;
74 amostras da classe 0 foram incorretamente classificadas como classe 1;
69 amostras da classe 1 foram incorretamente classificadas como classe 0;
1.141 amostras da classe 1 foram corretamente classificadas como classe 1.

O modelo classificou corretamente:

1.754 amostras

de um total de:

1.897 amostras

resultando em:

92,46% de accuracy.

A quantidade total de erros foi:

143 amostras

correspondendo a aproximadamente 7,54% do conjunto de teste.

Um aspecto positivo é que a quantidade de erros entre as classes é bastante semelhante:

| Tipo de erro        | Quantidade |
| ------------------- | ---------: |
| Classe 0 → Classe 1 |     **74** |
| Classe 1 → Classe 0 |     **69** |

Essa proximidade indica que os erros não estão fortemente concentrados em uma única classe.

5. Precision

A precision indica, entre as amostras que o modelo classificou como pertencentes a determinada classe, quantas realmente pertenciam a essa classe.

Os resultados foram:

| Classe              |  Precision |
| ------------------- | ---------: |
| Classe 0            |   **0,90** |
| Classe 1            |   **0,94** |
| **Precision macro** | **≈ 0,92** |


O modelo apresenta boa capacidade de realizar classificações positivas corretamente para ambas as classes.

A diferença entre as classes é relativamente pequena, com vantagem para a classe 1.

6. Recall

O recall mede a capacidade do modelo de identificar corretamente as amostras que realmente pertencem a determinada classe.

Os resultados foram:

| Classe           |     Recall |
| ---------------- | ---------: |
| Classe 0         |   **0,89** |
| Classe 1         |   **0,94** |
| **Recall macro** | **≈ 0,92** |


O recall de 0,89 para a classe 0 indica que aproximadamente 89% das amostras pertencentes à classe 0 foram corretamente identificadas.

Para a classe 1, o recall de 0,94 indica que aproximadamente 94% das amostras pertencentes à classe 1 foram corretamente identificadas.

A diferença entre os recalls mostra que a classe 0 apresenta maior dificuldade de classificação para o modelo.

7. F1-score

O F1-score combina precision e recall em uma única métrica, sendo especialmente útil quando existe interesse em equilibrar os dois tipos de desempenho.

Os resultados foram:

| Classe       |   F1-score |
| ------------ | ---------: |
| Classe 0     |   **0,90** |
| Classe 1     |   **0,94** |
| **F1 macro** | **≈ 0,92** |


O F1-score de 0,90 para a classe 0 e 0,94 para a classe 1 demonstra que o modelo apresenta desempenho elevado em ambas as classes.

O F1 macro é particularmente importante neste experimento porque calcula a média do desempenho das classes atribuindo o mesmo peso a cada uma delas.

Assim, o valor próximo de 0,92 demonstra que o desempenho do modelo permanece elevado mesmo quando as duas classes recebem o mesmo peso na avaliação.

8. Accuracy

A accuracy obtida no conjunto de teste foi:

Accuracy = 0,9246

Ou seja, aproximadamente 92,46% das amostras foram classificadas corretamente.

Embora seja uma métrica importante, a accuracy não deve ser utilizada isoladamente neste experimento.

O conjunto apresenta distribuição desigual entre as classes:

| Classe    |  Amostras | Proporção |
| --------- | --------: | --------: |
| Classe 0  |       687 |    36,22% |
| Classe 1  |     1.210 |    63,78% |
| **Total** | **1.897** |  **100%** |


Consequentemente, uma avaliação baseada exclusivamente em accuracy poderia ocultar diferenças de desempenho entre as classes.

Por esse motivo, F1 macro, precision macro, recall macro e as métricas individuais por classe são mais informativas para a análise do modelo.

9. Comparação entre validação cruzada e teste

Uma característica importante dos resultados é a proximidade entre o desempenho observado durante a validação cruzada e aquele obtido no conjunto de teste.

Validação cruzada
| Métrica         |  Resultado |
| --------------- | ---------: |
| Accuracy        |     0,9246 |
| Precision macro |     0,9174 |
| Recall macro    |     0,9202 |
| **F1 macro**    | **0,9187** |

Conjunto de teste
| Métrica         |  Resultado |
| --------------- | ---------: |
| Accuracy        | **0,9246** |
| Precision macro |     ≈ 0,92 |
| Recall macro    |     ≈ 0,92 |
| **F1 macro**    | **≈ 0,92** |

A proximidade entre os resultados não indica evidência de uma queda significativa de desempenho ao aplicar o modelo a dados não utilizados durante o treinamento.

Portanto, os resultados sugerem que o Random Forest apresentou boa capacidade de generalização para o conjunto de teste.

10. Influência de n_estimators

A análise das configurações avaliadas pelo Grid Search mostrou:

| `n_estimators` | F1 macro médio | Melhor F1 macro |
| -------------: | -------------: | --------------: |
|            150 |       0,912176 |        0,916790 |
|        **200** |   **0,912264** |    **0,918670** |
|            300 |       0,912294 |        0,918622 |

Observa-se que o aumento do número de árvores produz apenas alterações marginais no desempenho.

O melhor resultado foi obtido com 200 árvores, enquanto 300 árvores apresentou resultado praticamente idêntico.

Dessa forma, a utilização de 200 árvores representa uma escolha adequada, pois proporciona desempenho equivalente ao obtido com 300 árvores com menor custo computacional.

Relação com a análise OOB

O resultado também explica por que o número ideal de árvores observado anteriormente pela análise OOB (Out-of-Bag) foi de aproximadamente 300, enquanto a validação cruzada selecionou 200.

A estimativa OOB e a validação cruzada são procedimentos diferentes e podem produzir pontos ótimos ligeiramente distintos.

Portanto, não existe contradição entre os resultados:

OOB: indicou aproximadamente 300 árvores;
Grid Search + validação cruzada: selecionou 200 árvores;
Diferença de desempenho entre 200 e 300: praticamente desprezível.

Nesse cenário, 200 árvores é uma escolha razoável por apresentar o melhor F1 macro observado e menor custo computacional.

11. Influência do class_weight

O melhor modelo utilizou:

class_weight='balanced_subsample'

Esse resultado é relevante devido à distribuição desigual das classes.

O parâmetro balanced_subsample realiza o balanceamento dos pesos das classes considerando as amostras utilizadas na construção de cada árvore da floresta.

A presença dessa configuração entre os melhores resultados indica que o tratamento do desbalanceamento contribuiu para obter um desempenho mais equilibrado entre as classes.

Isso é especialmente importante porque o objetivo não é apenas maximizar o número total de classificações corretas, mas também manter um bom desempenho na identificação da classe minoritária.

12. Análise dos erros

A matriz de confusão permite observar que foram cometidos:

| Tipo de resultado                  | Quantidade |
| ---------------------------------- | ---------: |
| Classe 0 corretamente classificada |    **613** |
| Classe 0 classificada como 1       |     **74** |
| Classe 1 classificada como 0       |     **69** |
| Classe 1 corretamente classificada |  **1.141** |
| **Total de acertos**               |  **1.754** |
| **Total de erros**                 |    **143** |

Os erros de classificação representam:

143 / 1.897 × 100 ≈ 7,54%

do conjunto de teste.

Um aspecto positivo é que a quantidade de erros entre as classes é bastante semelhante:

| Classe real |  Erros |
| ----------- | -----: |
| Classe 0    | **74** |
| Classe 1    | **69** |

Esse comportamento reforça a indicação de que o modelo não está simplesmente favorecendo a classe majoritária.

13. Principais conclusões

Os resultados obtidos permitem concluir que o Random Forest apresentou desempenho elevado na classificação realizada, alcançando aproximadamente 92,46% de accuracy no conjunto de teste e F1 macro próximo de 0,92.

O desempenho individual das classes também foi satisfatório. A classe 0 apresentou precision de 0,90, recall de 0,89 e F1-score de 0,90, enquanto a classe 1 apresentou precision de 0,94, recall de 0,94 e F1-score de 0,94.

A diferença entre as métricas das classes demonstra que a classe 0 é ligeiramente mais difícil de classificar. Entretanto, a diferença não é suficientemente grande para indicar um comportamento fortemente enviesado em favor da classe majoritária.

A matriz de confusão reforça essa conclusão. Foram observadas 74 classificações incorretas da classe 0 como classe 1 e 69 classificações incorretas da classe 1 como classe 0. Portanto, os erros estão relativamente equilibrados entre as classes.

O F1 macro de aproximadamente 0,92 é uma das principais evidências de desempenho do modelo, pois atribui o mesmo peso às duas classes. A precision macro e o recall macro, também próximos de 0,92, reforçam a indicação de que o modelo apresenta equilíbrio entre a capacidade de evitar classificações incorretas e a capacidade de identificar corretamente as amostras de cada classe.

O emprego de class_weight='balanced_subsample' também se mostrou adequado diante do desbalanceamento existente no conjunto de dados, contribuindo para que o modelo mantivesse desempenho satisfatório nas duas classes.

A comparação entre 150, 200 e 300 árvores mostrou que 200 árvores foi suficiente para atingir o melhor resultado, com F1 macro de 0,9187. O aumento para 300 árvores não produziu ganho relevante, apresentando F1 macro de 0,9186. Portanto, a utilização de 200 árvores oferece uma relação adequada entre desempenho preditivo e custo computacional.

Por fim, a proximidade entre os resultados da validação cruzada e do conjunto de teste indica que o modelo apresenta boa capacidade de generalização, não sendo observada uma redução relevante do desempenho quando aplicado a amostras não utilizadas no treinamento.

14. Síntese dos resultados

| Indicador                 |              Resultado |
| ------------------------- | ---------------------: |
| **Accuracy no teste**     |             **92,46%** |
| **F1 macro — CV**         |             **91,87%** |
| F1 classe 0               |                    90% |
| F1 classe 1               |                    94% |
| Precision classe 0        |                    90% |
| Precision classe 1        |                    94% |
| Recall classe 0           |                    89% |
| Recall classe 1           |                    94% |
| Precision macro — CV      |                 91,74% |
| Recall macro — CV         |                 92,02% |
| Número de árvores         |                **200** |
| Critério de divisão       |            **Entropy** |
| Balanceamento             | **Balanced Subsample** |
| Profundidade máxima       |                 **30** |
| Amostras no teste         |              **1.897** |
| Classificações corretas   |              **1.754** |
| Classificações incorretas |                **143** |
| Taxa de acerto            |             **92,46%** |
| Taxa de erro              |              **7,54%** |

15. Conclusão geral

O Random Forest demonstrou desempenho consistente e equilibrado na classificação das duas classes, alcançando accuracy de 92,46% e F1 macro próximo de 0,92. As métricas de precision, recall e F1-score por classe indicam que o modelo não depende exclusivamente da classe majoritária para obter seu desempenho, apresentando resultados satisfatórios também para a classe minoritária. A matriz de confusão confirma esse comportamento, uma vez que os erros de classificação foram relativamente equilibrados entre as duas classes. A utilização de balanced_subsample mostrou-se adequada ao cenário de desbalanceamento, enquanto 200 árvores proporcionou o melhor compromisso entre desempenho e custo computacional. Em conjunto, os resultados obtidos indicam que o modelo possui boa capacidade de generalização e apresenta-se como uma abordagem adequada para o problema de classificação investigado.


