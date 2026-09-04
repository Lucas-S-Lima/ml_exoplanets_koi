# ml_exoplanets_decisions_001

# TCC: Classificação de Exoplanetas Utilizando Machine Learning Aplicado aos Dados da Missão Kepler

## Documento de Decisão #001

**Assunto:** Seleção inicial das features da base Cumulative KOI (NASA Exoplanet Archive)

**Data:** 28/06/2026

---

# 1. Contexto

O objetivo deste trabalho é desenvolver e comparar modelos de Machine Learning capazes de classificar candidatos a exoplanetas utilizando dados da missão espacial Kepler, disponibilizados pelo NASA Exoplanet Archive.

A base escolhida foi a **Cumulative KOI (Kepler Objects of Interest)**, que consolida os resultados das observações realizadas durante os quarters Q1-Q17 da missão Kepler.

A tabela Cumulative KOI possui mais de uma centena de atributos. Entretanto, grande parte deles corresponde a metadados administrativos, identificadores ou variáveis derivadas do processo de validação realizado pela própria NASA.

Dessa forma, foi realizada uma seleção inicial de atributos baseada em três critérios:

* relevância física;
* potencial discriminatório para algoritmos de classificação;
* possibilidade de reutilização futura em outras missões, como TESS.

---

# 2. Variável alvo

## Coluna selecionada

| Coluna            | Descrição                                                              |
| ----------------- | ---------------------------------------------------------------------- |
| `koi_disposition` | Classificação oficial do objeto realizada pela equipe da missão Kepler |

### Classes existentes

* CONFIRMED
* FALSE POSITIVE
* CANDIDATE

### Decisão

Inicialmente, serão utilizadas apenas as classes:

* CONFIRMED
* FALSE POSITIVE

Os registros classificados como CANDIDATE serão removidos da etapa de treinamento.

### Justificativa

A remoção da classe CANDIDATE transforma o problema em uma classificação binária supervisionada, reduzindo a complexidade inicial do modelo e facilitando a avaliação comparativa dos algoritmos.

---

# 3. Features selecionadas

## 3.1 koi_period

### Significado

Período orbital do objeto candidato, medido em dias.

### Justificativa

O período orbital é uma das principais características físicas utilizadas para caracterização de exoplanetas e apresenta forte capacidade discriminatória.

---

## 3.2 koi_duration

### Significado

Duração do trânsito do objeto em horas.

### Justificativa

A duração do trânsito fornece informações importantes sobre geometria orbital, tamanho relativo dos corpos e possíveis inconsistências observacionais.

---

## 3.3 koi_depth

### Significado

Profundidade do trânsito em partes por milhão (ppm).

### Justificativa

A profundidade do trânsito está diretamente relacionada ao tamanho relativo entre planeta e estrela, sendo uma das variáveis mais relevantes para identificação de exoplanetas.

---

## 3.4 koi_prad

### Significado

Raio estimado do planeta em raios terrestres.

### Justificativa

O tamanho do objeto é um dos principais fatores utilizados para diferenciar exoplanetas de falsos positivos.

---

## 3.5 koi_insol

### Significado

Fluxo de radiação estelar incidente no planeta, expresso em unidades terrestres.

### Justificativa

Permite caracterizar o ambiente orbital e pode contribuir para a separação entre diferentes tipos de objetos.

---

## 3.6 koi_teq

### Significado

Temperatura de equilíbrio estimada do planeta.

### Justificativa

Representa uma característica física importante e amplamente utilizada em estudos de caracterização planetária.

---

## 3.7 koi_impact

### Significado

Parâmetro de impacto do trânsito, representando a distância relativa entre o centro do trânsito e o centro da estrela.

### Justificativa

Fornece informações geométricas relevantes para distinguir trânsitos reais de eventos espúrios.

---

## 3.8 koi_model_snr

### Significado

Signal-to-noise ratio (SNR) do evento detectado.

### Justificativa

Eventos reais normalmente apresentam padrões distintos de relação sinal-ruído quando comparados a falsos positivos.

---

## 3.9 koi_steff

### Significado

Temperatura efetiva da estrela hospedeira, medida em Kelvin.

### Justificativa

As características da estrela influenciam diretamente os parâmetros observados do trânsito.

---

## 3.10 koi_slogg

### Significado

Logaritmo da gravidade superficial da estrela.

### Justificativa

Permite caracterizar o tipo e a estrutura da estrela hospedeira.

---

## 3.11 koi_srad

### Significado

Raio da estrela hospedeira em raios solares.

### Justificativa

O tamanho da estrela afeta diretamente as medidas de profundidade e duração do trânsito.

---

## 3.12 koi_kepmag

### Significado

Magnitude aparente da estrela no filtro fotométrico do telescópio Kepler.

### Justificativa

A luminosidade observada influencia a qualidade do sinal detectado e pode impactar a classificação.

---

# 4. Features descartadas

Foram descartadas inicialmente as seguintes categorias de atributos:

## Identificadores

* kepid
* kepoi_name
* kepler_name
* rowid

### Justificativa

Não possuem significado físico e podem introduzir overfitting.

---

## Metadados administrativos

* koi_vet_date
* koi_vet_stat
* koi_tce_delivname
* koi_trans_mod

### Justificativa

Representam informações administrativas do processo de catalogação.

---

## Variáveis de data leakage

* koi_fpflag_nt
* koi_fpflag_ss
* koi_fpflag_co
* koi_fpflag_ec
* koi_score

### Justificativa

Estas variáveis foram utilizadas pela própria equipe científica durante o processo de validação dos candidatos e poderiam fazer com que o modelo aprendesse diretamente a resposta correta, comprometendo a validade científica do estudo.

---

## Strings descritivas

* ra_str
* dec_str
* koi_pdisposition

### Justificativa

Não agregam informação física relevante para a etapa inicial de modelagem.

---

# 5. Conjunto inicial de features

```python
FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_insol",
    "koi_teq",
    "koi_impact",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]

TARGET = "koi_disposition"
```

# Renomeação de campos para melhor entendimento

| Campo Original | Nome Renomeado | Descrição | Unidade |
|---|---|---|---|
| `koi_period` | `orbital_period_days` | Período orbital do planeta | dias |
| `koi_duration` | `transit_duration_hours` | Duração do trânsito observado | horas |
| `koi_depth` | `transit_depth_ppm` | Profundidade do trânsito (queda de brilho da estrela) | ppm |
| `koi_prad` | `planet_radius_earth` | Raio do planeta | raios terrestres |
| `koi_insol` | `insolation_flux_earth` | Fluxo de radiação recebido pelo planeta | fluxo terrestre |
| `koi_teq` | `equilibrium_temperature_k` | Temperatura de equilíbrio do planeta | Kelvin |
| `koi_impact` | `impact_parameter` | Distância projetada centro estelar–planeta no trânsito, normalizada pelo raio estelar | adimensional |
| `koi_model_snr` | `transit_signal_to_noise` | Razão sinal-ruído do trânsito detectado | adimensional |
| `koi_steff` | `stellar_effective_temperature_k` | Temperatura efetiva da estrela hospedeira | Kelvin |
| `koi_slogg` | `stellar_surface_gravity` | Gravidade superficial da estrela hospedeira | log(cm/s²) |
| `koi_srad` | `stellar_radius_solar` | Raio da estrela hospedeira | raios solares |
| `koi_kepmag` | `kepler_magnitude` | Magnitude aparente da estrela no fotômetro Kepler | mag |
| `koi_disposition` | `label` | Classificação do objeto (target do modelo) | `CONFIRMED` / `CANDIDATE` / `FALSE POSITIVE` |

---

## Identificadores

* kepid
* kepler_name
* rowid

### Justificativa

Não possuem significado físico e podem introduzir overfitting.

---

## Metadados administrativos

* kepoi_name
* koi_vet_date
* koi_vet_stat
* koi_tce_delivname
* koi_trans_mod

### Justificativa

Representam informações administrativas do processo de catalogação. **Nota:** `kepoi_name` é mantido no dataset para rastreabilidade e identificação única dos objetos, mas não é utilizado como feature no modelo.

---

# 6. Decisão final

Foi definido que a primeira versão do modelo utilizará exclusivamente variáveis com significado físico direto, evitando atributos administrativos, identificadores e variáveis com potencial vazamento de informação (data leakage).

Esta decisão busca garantir:

* interpretabilidade científica;
* reprodutibilidade;
* generalização para futuras missões;
* comparabilidade entre diferentes algoritmos de Machine Learning.
