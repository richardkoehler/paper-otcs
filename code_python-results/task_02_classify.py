"""Apply pre-trained model to hold-out data for quantification results."""

import json
from pathlib import Path

import pandas as pd

import filetools
import classification


def main() -> None:
    """Main function of this script."""
    for compound in ["efudix", "cisplatin", "veregen"]:
        data_qupath = (
            Path(__file__).resolve().parents[1] / "data" / "nuclei_stardist_2022-01-23"
        )
        results_dir = data_qupath / f"Detections_results_{compound}"
        if not results_dir.is_dir():
            raise ValueError(f"Directory does not exist:\n{results_dir}")
        output_dir = data_qupath / f"Results_{compound}_EP"
        output_dir.mkdir(exist_ok=True)

        detection_files = filetools.find_files(
            path=results_dir, suffix="json", prefix="detections", verbose=True
        )

        treatment_file = f"{compound}.csv"
        treatment_map: pd.DataFrame = pd.read_csv(  # type: ignore
            data_qupath / treatment_file,
            index_col=0,
        )

        # Reload trained model
        model_path = data_qupath / "xgb_EP" / "model_final"
        model, features_used = classification.load_model_xgb(path=model_path)

        with open(model_path / "class_definitions.json", "r", encoding="utf-8") as file:
            class_ids = json.load(file)

        classification.classify_detections(
            files=detection_files,
            model=model,
            class_ids=class_ids,
            treatment_map=treatment_map,
            features_used=features_used,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
