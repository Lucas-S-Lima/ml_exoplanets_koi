import requests
import pandas as pd
from io import StringIO
import time

query = """
select
    kepoi_name,
    koi_disposition,
    koi_period,
    koi_duration,
    koi_depth,
    koi_prad,
    koi_insol,
    koi_teq,
    koi_impact,
    koi_model_snr,
    koi_steff,
    koi_slogg,
    koi_srad,
    koi_kepmag
from cumulative
"""

url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


max_retries = 3
timeout = 120

print("Iniciando download da NASA Exoplanet Archive\n")

session = requests.Session()

for attempt in range(1, max_retries + 1):
    start_time = time.time()

    try:
        print(f"Tentativa {attempt}/{max_retries}")

        response = session.get(url, params={"query": query, "format": "csv"}, timeout=timeout)

        print("Status HTTP:", response.status_code)
        response.raise_for_status()

        elapsed_time = time.time() - start_time
        size_kb = len(response.content) / 1024

        print(f"Tempo de resposta: {elapsed_time:.2f} segundos")
        print(f"Tamanho do payload: {size_kb:.2f} KB")

        df = pd.read_csv(StringIO(response.content.decode("utf-8")))

        print("\nDataset carregado com sucesso!")
        print("Shape:", df.shape)
        print("\nPreview:")
        print(df.head())

        output_file = "data/cumulative_koi.csv"
        df.to_csv(output_file, index=False)

        print(f"\nArquivo salvo em: {output_file}")
        break

    except requests.exceptions.ChunkedEncodingError:
        print("Conexão interrompida (ChunkedEncodingError)")
        print("Tentando novamente em 3 segundos...\n")
        time.sleep(3)

    except requests.exceptions.Timeout:
        print("Timeout: servidor demorou demais para responder")
        print("Tentando novamente...\n")

    except requests.exceptions.RequestException as e:
        print("Erro HTTP geral:", e)
        break

else:
    print("Falha após múltiplas tentativas. NASA API instável ou rede com problema.")
