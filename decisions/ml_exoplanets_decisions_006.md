# ml_exoplanets_decisions_002

# TCC: Classificação de Exoplanetas Utilizando Machine Learning Aplicado aos Dados da Missão Kepler

## Documento de Decisão #002

**Assunto:** Tratamento logarítmico de features com distribuição de cauda longa na Regressão Logística

**Data:** 27/08/2026

---

## 1. Contexto

Durante a análise dos resultados da Regressão Logística final (`C=3000`, `l1_ratio=0.0`, `solver=lbfgs`, `max_iter=2000`, `random_state=42`), identificou-se um comportamento anômalo na saída linear do modelo (`z = decision_function`) sobre o conjunto de teste: valores variando de faixas normais até **z ≈ 2667**, muito acima do necessário para saturar a função sigmoide (que já satura em torno de |z| > 10).

Essa saída extrema estava associada a coeficientes com magnitudes bastante díspares entre si:

| Feature (índice) | Coeficiente |
|---|---|
| `planet_radius_earth` (3) | **171,41** |
| `insolation_flux_earth` (4) | **49,79** |
| demais 10 features | entre 0,057 e 9,20 |

Intercept: 11,503 — acurácia de teste: 85,55%.

## 2. Conceitos utilizados neste documento

Antes de detalhar o diagnóstico, seguem as definições dos termos técnicos usados ao longo deste documento, para leitores sem familiaridade prévia com eles.

**z (saída linear / *decision function*)**

Na Regressão Logística, antes de o modelo gerar uma probabilidade, ele calcula uma combinação linear das features:

```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

onde `β₀` é o intercepto, cada `βᵢ` é o coeficiente aprendido para a feature `xᵢ`. Esse `z` (chamado de *decision function* no scikit-learn) pode assumir qualquer valor real, positivo ou negativo, sem limite superior ou inferior. É só depois, ao aplicar a função sigmoide `σ(z) = 1 / (1 + e⁻ᶻ)`, que esse valor é "espremido" para o intervalo entre 0 e 1, virando uma probabilidade. A sigmoide já satura (fica achatada em 0 ou 1, sem distinção prática de confiança) para `|z|` acima de aproximadamente 10 — por isso um `z` chegando a milhares de unidades indica um problema no modelo, não uma classificação "mais confiante".

**Assimetria / skew**

É uma medida numérica de quão distante uma distribuição de valores está de ser simétrica (em forma de sino, com valores espalhados igualmente para os dois lados da média). Um skew próximo de 0 indica simetria. Um skew alto e positivo indica que a maioria dos valores está concentrada numa faixa relativamente baixa, mas existe uma "cauda" de valores bem mais altos puxando a distribuição para a direita — é o caso típico de grandezas físicas multiplicativas (como o raio de um planeta), em que a maioria dos casos é "normal" mas alguns poucos casos extremos são ordens de grandeza maiores. Como regra prática, valores de skew acima de 1 (em módulo) já são considerados relevantes; os valores acima de 50 observados neste projeto (ver Seção 6) são extremamente atípicos.

**Transformação monotônica**

É qualquer transformação matemática que preserva a ordem relativa dos valores: se `a > b` antes da transformação, então `f(a) > f(b)` depois dela também — a transformação apenas "reescala" os valores, sem trocar a ordem entre eles. O logaritmo é um exemplo de transformação monotônica. Essa propriedade é o que explica por que modelos baseados em árvore de decisão (Random Forest, Gradient Boosting) não são afetados pela transformação logarítmica: como esses modelos decidem seus cortes (*splits*) comparando valores (ex: "`planet_radius_earth` > 5?"), e a ordem entre os valores não muda com uma transformação monotônica, os cortes encontrados pela árvore são equivalentes antes e depois do log — só o valor numérico do limiar muda de escala. Já a Regressão Logística é sensível à transformação porque combina as features numa soma linear (o `z` definido acima), então a *escala* de cada feature afeta diretamente o peso que ela recebe na decisão.

**log1p (log(1 + x))**

É a função `log1p(x) = log(1 + x)`, usada para comprimir a cauda longa de uma distribuição assimétrica. Foi escolhida no lugar do logaritmo tradicional (`log(x)`) porque este último não é definido para `x = 0` — resultaria em erro ou `-infinito`. Como algumas das features deste projeto podem conter valores iguais a zero, o `log1p` resolve isso: quando `x = 0`, o resultado é `log(1) = 0`, sem erro. O efeito prático é reduzir a distância relativa entre valores muito grandes (ex: a diferença entre um valor de 1.000 e um de 10.000 fica proporcionalmente muito menor depois do log do que a diferença entre 1 e 10), sem eliminar a informação de que um valor é maior que o outro — é justamente essa compressão que impede que os poucos outliers extremos dominem sozinhos o coeficiente do modelo linear.

## 3. Diagnóstico

Descartou-se a hipótese de vazamento de dado: as 12 features do modelo (`orbital_period_days`, `transit_duration_hours`, `transit_depth_ppm`, `planet_radius_earth`, `insolation_flux_earth`, `equilibrium_temperature_k`, `impact_parameter`, `transit_signal_to_noise`, `stellar_effective_temperature_k`, `stellar_surface_gravity`, `stellar_radius_solar`, `kepler_magnitude`) são medições físicas legítimas do trânsito e da estrela, sem nenhuma coluna derivada do processo de vetting que gera o rótulo `koi_disposition`.

A causa identificada foi a distribuição de cauda longa de `planet_radius_earth` e `insolation_flux_earth` — grandezas físicas multiplicativas por natureza, em que amostras `FALSE POSITIVE` (tipicamente binárias eclipsantes confundidas com trânsitos planetários) apresentam valores ordens de grandeza acima da maioria. Mesmo após o `StandardScaler` aplicado no pipeline (ver `x_koi_scaling_006`), esses outliers extremos mantinham valores padronizados muito altos, o que levava a Regressão Logística — sem regularização efetiva, dado o `C=3000` encontrado no GridSearchCV — a atribuir coeficientes desproporcionais a essas duas features para acomodá-los.

Essa distorção é específica de modelos lineares. Modelos baseados em árvore (Random Forest, Gradient Boosting) não são afetados pelo mesmo problema, pois decidem seus splits comparando valores (`feature > threshold`) e são, portanto, invariantes a transformações monotônicas como o log — a ordem relativa dos valores não muda, apenas a escala.

## 4. Decisão

Aplicar a transformação `log1p` às seguintes features antes da padronização com `StandardScaler`:

- `orbital_period_days`
- `transit_depth_ppm`
- `planet_radius_earth`
- `insolation_flux_earth`

As demais 8 features permanecem sem transformação logarítmica, por não apresentarem o mesmo padrão de assimetria acentuada.

## 5. Impacto nos modelos já treinados

| Modelo | Necessidade de novo tuning (GridSearchCV) |
|---|---|
| Regressão Logística | **Sim** — é o modelo afetado pela distorção; deve ser re-tunado e reanalisado por completo |
| Random Forest | **Não** — hiperparâmetros já definidos (`n_estimators=200`, `criterion=entropy`, `class_weight=balanced_subsample`, `max_depth=30`, `min_samples_leaf=1`, `max_features=sqrt`) devem ser mantidos; invariante a transformações monotônicas (ver Seção 2) |
| SVM (kernel RBF) | A confirmar — sensível à escala, mas não necessariamente à assimetria; avaliar caso a caso |
| Gradient Boosting | Mesma justificativa do Random Forest (baseado em árvore) — não deve exigir novo tuning |

## 6. Resultados obtidos

**Redução da assimetria (skew) após o log1p**, medida em `x_koi_scaling_006.py`:

| Feature | Skew antes | Skew depois |
|---|---|---|
| `orbital_period_days` | 3,47 | 0,61 |
| `transit_depth_ppm` | 5,37 | 1,20 |
| `planet_radius_earth` | 53,13 | 1,48 |
| `insolation_flux_earth` | 50,80 | 0,19 |

Os valores de skew de `planet_radius_earth` e `insolation_flux_earth` antes da transformação (53,13 e 50,80) estavam muito acima do que se considera assimetria relevante (>1), confirmando que essas eram, de fato, as duas features responsáveis pelos coeficientes desproporcionais (171,41 e 49,79) observados na Regressão Logística original. Após o `log1p`, ambas caíram para valores próximos de 1 ou abaixo disso — uma redução de mais de 30 vezes no caso de `planet_radius_earth`.

**Confirmação da invariância do Random Forest**: o modelo foi re-treinado sobre os dados transformados, mantendo os mesmos hiperparâmetros já definidos (sem novo `GridSearchCV`), e obteve acurácia de teste de **92,57%** (usando o arquivo `exoplanets_split.pkl` já com o log-transform) — praticamente idêntica à acurácia original de 92,46% obtida antes da transformação logarítmica. Isso confirma empiricamente a propriedade descrita na Seção 2: transformações monotônicas como o log não alteram o comportamento de modelos baseados em árvore de decisão, pois a ordem relativa dos valores — o único aspecto relevante para os cortes (*splits*) do modelo — permanece a mesma.

## 7. Próximos passos

1. ~~Rodar `x_koi_scaling_006.py` e conferir a redução da assimetria (`skew()`) nas 4 colunas selecionadas.~~ — concluído, ver Seção 6.
2. Re-treinar a Regressão Logística sobre os dados transformados e comparar coeficientes, range de `z` e acurácia de teste com os valores deste documento (171,41 / 49,79 / 85,55%).

   **Resultado (concluído):** intercept caiu de 11,503 para **3,083**; o maior coeficiente (`planet_radius_earth`) caiu de 171,41 para **9,25**, ficando na mesma ordem de grandeza das demais 11 features; o `z` no conjunto de teste passou de um range de -3,12 a **2667,31** para **-5,35 a 49,91** — uma redução de mais de 50 vezes no valor máximo. A acurácia de teste teve uma queda pequena, de 0,8555 para **0,8471** (menos de 1 ponto percentual), e o F1-score por classe se manteve praticamente igual (0,80/0,88 contra 0,81/0,88 antes). Conclui-se que o modelo deixou de depender de poucos outliers extremos para sustentar sua confiança, sem perda relevante de poder preditivo.
3. ~~Re-treinar o Random Forest uma vez, sem novo GridSearchCV, como checagem de sanidade da invariância a transformações monotônicas.~~ — concluído, ver Seção 6 (92,57% no arquivo corrigido, praticamente idêntico ao original).
4. ~~Formalizar o fechamento deste documento após a reanálise da Regressão Logística.~~ — concluído nesta atualização.

## 8. Conclusão

A causa raiz do comportamento anômalo da Regressão Logística (coeficientes desproporcionais e saída `z` saturando em milhares) foi confirmada como sendo a distribuição de cauda longa de `planet_radius_earth` e `insolation_flux_earth`, e não vazamento de dado. A aplicação de `log1p` a essas duas features, mais `orbital_period_days` e `transit_depth_ppm`, resolveu o problema: os coeficientes ficaram equilibrados, o `z` voltou a uma faixa razoável, e a acurácia se manteve praticamente estável. O Random Forest, como esperado por sua invariância a transformações monotônicas, não foi afetado pela mudança. A decisão de aplicar o log-transform nessas quatro features é considerada validada e encerrada.