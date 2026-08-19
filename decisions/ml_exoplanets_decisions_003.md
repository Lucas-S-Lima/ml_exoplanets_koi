# Relatório de Decisão — Estratégia de Rotulagem para Treino e Teste do Modelo

**Projeto:** Classificação de Exoplanetas (KOI — Kepler Objects of Interest)
**Data:** 01/08/2026

## Documento de Decisão #002

## 1. Contexto

O dataset tratado (`cumulative_koi_treated.csv`) possui a coluna `label` (originalmente `koi_disposition`) como variável alvo, contendo três classes possíveis:

- `CONFIRMED` — objeto confirmado como exoplaneta
- `CANDIDATE` — objeto ainda sem confirmação definitiva
- `FALSE POSITIVE` — objeto descartado como exoplaneta

Era necessário definir quais dessas classes seriam utilizadas nas etapas de treino e teste do modelo de classificação.

## 2. Problema

A classe `CANDIDATE` representa um estado de **incerteza** — objetos que ainda não passaram por confirmação ou descarte definitivo pela comunidade científica. Incluir essa classe diretamente no treino/teste levanta dois problemas:

1. **Ruído no rótulo:** `CANDIDATE` não é uma categoria com características físicas próprias; mistura objetos que futuramente serão confirmados com objetos que futuramente serão descartados. O modelo não teria um padrão consistente para aprender.
2. **Desalinhamento com o objetivo do projeto:** o propósito prático do modelo é justamente auxiliar a triagem de candidatos ainda não confirmados — usá-los como dado de treino contradiz esse objetivo.

## 3. Decisão

Foi decidido:

- **Treino e teste do modelo:** utilizar exclusivamente os registros rotulados como `CONFIRMED` e `FALSE POSITIVE`, tratando o problema como uma **classificação binária**.
- **Registros `CANDIDATE`:** mantidos separados do conjunto de treino/teste. Serão utilizados **após** o treinamento e validação do modelo, como entrada para geração de novas predições.

## 4. Justificativa

- É a abordagem cientificamente mais sólida e mais comum na literatura de classificação de KOIs, pois evita treinar o modelo com um rótulo que representa "ainda não sabemos" em vez de uma característica real do objeto.
- Simula o cenário real de uso do modelo: dado um objeto ainda não confirmado, estimar a probabilidade de ser um exoplaneta real ou um falso positivo.
- Evita introduzir ruído estatístico numa classe que não possui padrão físico próprio.

## 5. Impacto na implementação

- O split treino/teste (`train_test_split`) usará `stratify` sobre o target para preservar a proporção entre `CONFIRMED` e `FALSE POSITIVE`.
- O `LabelEncoder` do target será ajustado (`fit`) apenas sobre as duas classes utilizadas.
- Após treino e validação do modelo, os registros `CANDIDATE` serão submetidos ao modelo treinado (`predict`) para gerar uma classificação estimada, sem participarem das métricas de avaliação do modelo.

## 6. Alternativa descartada

Classificação multiclasse (`CONFIRMED` / `CANDIDATE` / `FALSE POSITIVE`) foi considerada e descartada, por exigir que o modelo aprendesse um padrão consistente para uma classe (`CANDIDATE`) que representa um estado transitório de incerteza, e não uma categoria fisicamente distinta.