"""Train XGBoost Classifier and write cross-validation results."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

import filetools
import classification


def main() -> None:
    """Main function of this script."""
    data_qupath = (
        Path(__file__).resolve().parents[1] / "data" / "nuclei_stardist_2022-01-23"
    )
    for researcher in ["RK", "EP"]:
        training_dir = data_qupath / f"Training_{researcher}"
        output_dir = data_qupath / f"xgb_{researcher}"
        output_dir.mkdir(exist_ok=True)

        # Find files and load data
        files = filetools.find_files(
            path=training_dir, suffix="csv", prefix="detections", verbose=True
        )
        data = pd.concat(
            [pd.read_csv(file) for file in files],  # type: ignore
            ignore_index=True,
        )

        # Write out cross-validation results
        classification.cross_validate_xgb(
            data=data, output_dir=output_dir, cross_validator=LeaveOneGroupOut()
        )
        if researcher == "EP":
            # Train final model
            classification.train_final_xgb(
                data=data, output_dir=output_dir / "model_final"
            )


if __name__ == "__main__":
    main()
