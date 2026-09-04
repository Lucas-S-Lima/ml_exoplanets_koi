import pandas as pd

COLUMN_RENAME = {
    "kepoi_name": "kepler_object_of_interest_name",
    "koi_disposition": "label",
    "koi_period": "orbital_period_days",
    "koi_duration": "transit_duration_hours",
    "koi_depth": "transit_depth_ppm",
    "koi_prad": "planet_radius_earth",
    "koi_insol": "insolation_flux_earth",
    "koi_teq": "equilibrium_temperature_k",
    "koi_impact": "impact_parameter",
    "koi_model_snr": "transit_signal_to_noise",
    "koi_steff": "stellar_effective_temperature_k",
    "koi_slogg": "stellar_surface_gravity",
    "koi_srad": "stellar_radius_solar",
    "koi_kepmag": "kepler_magnitude",
}


def rename_columns(input_path: str, output_path: str, column_map: dict) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df.rename(columns=column_map)

    if "label" in df.columns:
        cols = [c for c in df.columns if c != "label"] + ["label"]
        df = df[cols]

    df.to_csv(output_path, index=False)

    print(f"Arquivo salvo em: {output_path}")
    print(f"Shape: {df.shape}")
    print("\nPreview:")
    print(df.head())

    return df


if __name__ == "__main__":
    renamed_df = rename_columns(
        input_path="./data/cumulative_koi.csv",
        output_path="./data/cumulative_koi_renamed.csv",
        column_map=COLUMN_RENAME,
    )
