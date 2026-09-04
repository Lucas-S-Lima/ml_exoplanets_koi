import os
import time
from io import StringIO
import requests
import pandas as pd
from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings

logger = get_task_logger(__name__)


@shared_task(bind=True, name="download_exoplanet_data", max_retries=3)
def download_exoplanet_data(self):
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
    timeout = 120

    logger.info("Iniciando download da NASA Exoplanet Archive\n")

    attempt = self.request.retries + 1
    logger.info(f"Tentativa {attempt}/{self.max_retries}")

    session = requests.Session()
    start_time = time.time()

    try:
        response = session.get(url, params={"query": query, "format": "csv"}, timeout=timeout)

        logger.info(f"Status HTTP: {response.status_code}")
        response.raise_for_status()

        elapsed_time = time.time() - start_time
        size_kb = len(response.content) / 1024

        logger.info(f"Tempo de resposta: {elapsed_time:.2f} segundos")
        logger.info(f"Tamanho do payload: {size_kb:.2f} KB")

        df = pd.read_csv(StringIO(response.content.decode("utf-8")))

        logger.info("\nDataset carregado com sucesso!")
        logger.info(f"Shape: {df.shape}")

        output_dir = os.path.join(settings.BASE_DIR, "data")
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "cumulative_koi.csv")
        df.to_csv(output_file, index=False)

        logger.info(f"\nArquivo salvo em: {output_file}")

        return {"status": "success", "file": output_file}

    except requests.exceptions.ChunkedEncodingError as e:
        logger.warning("Conexão interrompida (ChunkedEncodingError)")
        logger.warning("Tentando novamente em 3 segundos...\n")

        raise self.retry(exc=e, countdown=3)

    except requests.exceptions.Timeout as e:
        logger.warning("Timeout: servidor demorou demais para responder")
        logger.warning("Tentando novamente...\n")

        raise self.retry(exc=e, countdown=1)

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro HTTP geral: {e}")
        raise
