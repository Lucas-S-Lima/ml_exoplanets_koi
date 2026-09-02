import os
import logging
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings


logger = logging.getLogger(__name__)

COLUMN_RENAME = {
    "kepoi_name":       "kepler_object_of_interest_name",
    "koi_disposition":  "label",
    "koi_period":       "orbital_period_days",
    "koi_duration":     "transit_duration_hours",
    "koi_depth":        "transit_depth_ppm",
    "koi_prad":         "planet_radius_earth",
    "koi_insol":        "insolation_flux_earth",
    "koi_teq":          "equilibrium_temperature_k",
    "koi_impact":       "impact_parameter",
    "koi_model_snr":    "transit_signal_to_noise",
    "koi_steff":        "stellar_effective_temperature_k",
    "koi_slogg":        "stellar_surface_gravity",
    "koi_srad":         "stellar_radius_solar",
    "koi_kepmag":       "kepler_magnitude",
}

class Command(BaseCommand):
    help = 'Processa o dataset bruto: renomeia colunas, limpa e trata os dados gerando o arquivo final exo_dataset.csv.'

    def handle(self, *args, **kwargs):
        data_dir = os.path.join(settings.BASE_DIR, 'data')
        input_file = os.path.join(data_dir, 'cumulative_koi.csv')
        output_file = os.path.join(data_dir, 'exo_dataset.csv')

        if not os.path.exists(input_file):
            logger.error(f"Arquivo bruto não encontrado: {input_file}. Execute a task de download primeiro.")
            return

        logger.info("Iniciando pipeline de processamento de dados...")

        try:
            # Etapa 1: Leitura e Renomeação de Colunas
            base_data = pd.read_csv(input_file)
            logger.info(f"Shape bruto original: {base_data.shape}")
            
            base_data = base_data.rename(columns=COLUMN_RENAME)

            if "label" in base_data.columns:
                cols = [c for c in base_data.columns if c != "label"] + ["label"]
                base_data = base_data[cols]

            logger.info(f"Colunas renomeadas e reorganizadas.")

            # Etapa 2: Limpeza e Tratamento Estatístico
            mask_inconsistente = (base_data['label'] == 'CANDIDATE') & (
                (base_data['transit_signal_to_noise'] == 0) |
                (base_data['transit_depth_ppm'] == 0) |
                (base_data['insolation_flux_earth'] == 0)
            )
            logger.info(f"Registros CANDIDATE com zero implausível removidos: {mask_inconsistente.sum()}")
            base_data = base_data.drop(base_data[mask_inconsistente].index)

            colunas_numericas_com_nulo = [
                'transit_depth_ppm', 'planet_radius_earth', 'insolation_flux_earth',
                'equilibrium_temperature_k', 'impact_parameter', 'transit_signal_to_noise',
                'stellar_effective_temperature_k', 'stellar_surface_gravity',
                'stellar_radius_solar', 'kepler_magnitude'
            ]

            logger.info("Preenchendo valores nulos com a mediana por classe...")
            for col in colunas_numericas_com_nulo:
                base_data[col] = base_data.groupby('label')[col].transform(
                    lambda x: x.fillna(x.median())
                )

            logger.info(f"Nulos restantes no total: {base_data.isnull().sum().sum()}")
            logger.info(f"Duplicatas completas: {base_data.duplicated().sum()}")
            
            if 'kepler_object_of_interest_name' in base_data.columns:
                duplicatas_kooi = base_data.duplicated(subset=["kepler_object_of_interest_name"]).sum()
                logger.info(f"Duplicatas por KOOI: {duplicatas_kooi}")

            logger.info(f"Shape final tratado: {base_data.shape}")
            
            distribuicao = base_data['label'].value_counts().to_dict()
            logger.info(f"Distribuição de classes: {distribuicao}")

            # Salvando o resultado final
            base_data.to_csv(output_file, index=False)
            logger.info(f"Sucesso! Dataset tratado salvo em: {output_file}")

        except Exception as e:
            logger.exception(f"Erro crítico durante o processamento do dataset: {e}")