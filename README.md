ML Exoplanets — Instruções para rodar o projeto
===============================================

Resumo
------
Projeto Django para exploração e processamento de dados de candidatos a exoplanetas, com scripts auxiliares de pré-processamento e divisão de dados em `scripts/` e `ml_core/`.

Requisitos
---------
- Python 3.8+
- `pip`

Instalação rápida
-----------------
1. Clone o repositório (se ainda não o fez):

```bash
git clone <repo_url>
cd ml_exoplanets
```

2. Crie e ative um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

Configuração e banco de dados
-----------------------------
O projeto usa `sqlite3` por padrão (`db.sqlite3` já pode existir). Rode as migrations e, se desejar, crie um superusuário:

```bash
python manage.py migrate
python manage.py createsuperuser  # opcional
```

Rodando a aplicação Django (desenvolvimento)
-------------------------------------------
Execute o servidor de desenvolvimento:

```bash
python manage.py runserver
```

Abra http://127.0.0.1:8000/ no navegador para acessar a aplicação.

Arquivos de mídia e estáticos
----------------------------
- Os arquivos estáticos estão em `static/` e os templates em `templates/`.
- Se for necessário, rode `collectstatic` antes de deploy (não obrigatório em desenvolvimento):

```bash
python manage.py collectstatic
```

Executando os scripts de dados e ML
----------------------------------
Alguns scripts de preparação e divisão de dados estão na pasta `scripts/` e `ml_core/`.
Exemplos:

```bash
python scripts/download_dataset_001.py
python scripts/rename_dataset_columns_003.py
python scripts/inconsistent_values_handling_004.py
python scripts/features_target_split_005.py
python scripts/x_koi_scaling_006.py
python scripts/y_koi_encoded_007.py
python scripts/training_test_split_008.py
```

Adapte argumentos conforme necessário; muitos scripts aceitam parâmetros (ver cabeçalho de cada script).

Testes
------
Rode a suíte de testes do Django:

```bash
python manage.py test
```

Estrutura principal do projeto
-----------------------------
- `manage.py` — utilitário do Django.
- `config/` — configurações do Django (`settings.py`, `urls.py`, etc.).
- `core/` — app principal do projeto (modelos, views, tarefas, templates).
- `scripts/`, `ml_core/` — scripts de processamento e tarefas de ML.
- `data/` — conjuntos de dados CSV usados no projeto.
- `decisions/`- decisões relevantes tomadas ao longo do projeto




