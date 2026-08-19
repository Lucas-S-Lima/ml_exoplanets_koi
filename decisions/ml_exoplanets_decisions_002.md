# ml_exoplanets_decisions_001

# TCC: Classificação de Exoplanetas Utilizando Machine Learning Aplicado aos Dados da Missão Kepler

## Documento de Decisão #002

**Assunto:** Normalização de valores inconsistentes.

**Data:** 12/07/2026

# Remover apenas zeros implausíveis em registros CANDIDATE
# Um candidato a planeta não deveria ter sinal/profundidade de trânsito zero —
# isso indica falha de medição, não uma característica válida da classe.
# (Zeros em FALSE POSITIVE foram mantidos: podem ser sinal legítimo de "não é planeta".)

# Outliers extremos NÃO são removidos
# Temperatura de equilíbrio, duração de trânsito e raio estelar extremos aparecem
# quase exclusivamente em FALSE POSITIVE/CANDIDATE (nenhum em CONFIRMED) — são um
# sinal característico da classe negativa, não erro de dado. Remover apagaria
# justamente a informação que ajuda o modelo a identificar falsos positivos.


# NOTA:
"Os valores extremos observados nas variáveis físicas (temperatura de equilíbrio, duração de trânsito, raio estelar) concentram-se nos registros classificados como FALSE POSITIVE, consistente com o processo de vetting automatizado do Kepler (Robovetter), que identifica sinais de trânsito espúrios frequentemente originados de binárias eclipsantes ou artefatos instrumentais (Thompson et al. 2018; Coughlin et al. 2016). Dessa forma, optou-se por preservar esses registros no conjunto de dados, tratando-os como informação discriminativa relevante para a tarefa de classificação, e não como inconsistência a ser removida."

# REFERÊNCIAS:
"""
Thompson et al. (2018) — "Planetary Candidates Observed by Kepler VIII: A Fully Automated Catalog With Measured Completeness and Reliability Based on Data Release 25", ApJS 235, 38. Este é o paper que descreve o Robovetter, o sistema que gerou as classificações CANDIDATE/FALSE POSITIVE do dataset DR25 (que muito provavelmente é a origem da sua base). Documenta as categorias de falso positivo (Not-Transit-Like, Stellar Eclipse, Centroid Offset, Ephemeris Match).

Coughlin et al. (2016) — "Planetary Candidates Observed by Kepler VII: The First Fully Uniform Catalog Based on the Entire 48-Month Dataset (Q1–Q17 DR24)", ApJS 224, 12. Descreve o processo de vetting uniforme, mencionado como base do Robovetter.

NASA Exoplanet Archive — documentação do catálogo KOI: https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html — fonte primária e citável diretamente, explica a origem de cada disposição (CANDIDATE, CONFIRMED, FALSE POSITIVE).

Han, Kamber & Pei — "Data Mining: Concepts and Techniques" (Morgan Kaufmann) — capítulo de data cleaning discute a diferença entre outlier como erro vs. outlier como informação relevante (ex: fraude, anomalias).

Aggarwal, C. C. — "Outlier Analysis" (Springer) — referência mais específica sobre outlier detection, inclusive a ideia de que em problemas de classificação supervisionada, outliers de uma classe podem ser exatamente o padrão que se quer aprender (não são "erro" nesse contexto).
"""
# CITAÇÃO:
"""
COUGHLIN, J. L. et al. Planetary Candidates Observed by Kepler. VII. The First
Fully Uniform Catalog Based on the Entire 48-month Dataset (Q1-Q17 DR24). The
Astrophysical Journal Supplement Series, v. 224, n. 1, p. 12, 2016.
Disponível em: https://arxiv.org/abs/1512.06149.

THOMPSON, S. E. et al. Planetary Candidates Observed by Kepler. VIII. A Fully
Automated Catalog With Measured Completeness and Reliability Based on Data
Release 25. The Astrophysical Journal Supplement Series, v. 235, n. 2, p. 38, 2018.
Disponível em: https://arxiv.org/abs/1710.06758.
"""

# OUTPUT:
"""
Shape original: (9564, 13)
Registros CANDIDATE com zero implausível: 2
Shape após remover: (9562, 13)
--------------------------------------------
Nulos restantes:
orbital_period_days                0
transit_duration_hours             0
transit_depth_ppm                  0
planet_radius_earth                0
insolation_flux_earth              0
equilibrium_temperature_k          0
impact_parameter                   0
transit_signal_to_noise            0
stellar_effective_temperature_k    0
stellar_surface_gravity            0
stellar_radius_solar               0
kepler_magnitude                   0
label                              0
--------------------------------------------
dtype: int64
Shape final: (9562, 13)

--------------------------------------------
Distribuição de classes:
label
FALSE POSITIVE    4839
CONFIRMED         2747
CANDIDATE         1976
Name: count, dtype: int64

Duplicatas: 0
"""
