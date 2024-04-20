"""Module for classification functions."""

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
import xgboost


def load_model_xgb(path) -> tuple[xgboost.XGBClassifier, list[str]]:
    """Load and return model and features used for model training."""
    if not path.is_dir():
        raise ValueError(f"Directory does not exist:\n{path}")

    model = xgboost.XGBClassifier()
    model.load_model(path / "model.json")
    features_used = list(model.get_booster().get_score(importance_type="weight"))
    return model, features_used


def classify_detections(
    files: list[Union[str, Path]],
    model: Any,
    class_ids: dict,
    treatment_map: pd.DataFrame,
    features_used: list[str],
    output_dir: Union[str, Path],
) -> None:
    "Classify object detections and save results."
    colors_rgb = {
        "Apoptosis/Necrosis": -16777216,
        "SiHa": -65281,
        "Keratinocytes": -16711681,
    }

    count_dicts = []

    for file in files:
        print(f"Using file:\n{file}")
        file = str(file)
        df = pd.read_json(file)
        feature_list = []
        for idx, row in df.iterrows():
            feature_list.append(
                [item["value"] for item in row["properties"]["measurements"]]
            )
        features = pd.DataFrame(
            data=feature_list,
            columns=[
                item["name"] for item in df.iloc[-1]["properties"]["measurements"]
            ],
        )

        del_cols = [col for col in features.columns if col not in features_used]
        features = features.drop(columns=del_cols)

        idx_end = file.find("detections")
        annotations_file = file[:idx_end] + "annotations.csv"
        annotations: pd.DataFrame = pd.read_csv(  # type: ignore
            annotations_file, delimiter=","
        )
        annotations = annotations.drop(
            np.where(annotations["Name"] == "ColorDeconvolution")[0],
            axis=0,
        )
        roi = annotations["Area µm^2"].sum() / 1000000

        result_pred = model.predict(features)

        with open(file, "r", encoding="utf-8") as in_file:
            json_data = json.load(in_file)
        for idx, class_id in enumerate(result_pred):
            json_data[idx]["properties"]["classification"] = {
                "name": class_ids[str(class_id)],
                "colorRGB": colors_rgb[class_ids[str(class_id)]],
            }

        # Save classifications
        outpath = Path(output_dir) / (Path(file).name[:-5] + "_new.json")
        with open(outpath, "w", encoding="utf-8") as out_file:
            json.dump(json_data, out_file)

        unique, counts = np.unique(result_pred, return_counts=True)
        counts = dict(zip(unique, counts / roi))

        # Specimen
        ind_beg = file.find("#")
        specimen = file[ind_beg + 1 : ind_beg + 4]
        counts["Specimen"] = specimen

        # Level
        ind_ebene = file.find("Ebene")
        ebene = file[ind_ebene + 5 : ind_ebene + 7]
        counts["Ebene"] = ebene

        # Dose and duration
        counts["Dose"], counts["Treatment Duration"] = treatment_map.loc[
            int(specimen), :
        ]
        count_dicts.append(counts)

    counts = pd.DataFrame(count_dicts).rename(
        columns={int(key): val for key, val in class_ids.items()}
    )
    counts.to_csv(Path(output_dir) / "counts.csv")


def balance_samples(
    data: np.ndarray, labels: np.ndarray, method: str = "oversample"
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Balance class sizes to create equal class distributions.

    Parameters
    ----------
    data : numpy array
        Feature array of shape (n_features, n_samples)
    labels : numpy array
        Array of class distribution of shape (n_samples, )
    method : {'oversample', 'undersample', 'weight'}
        Method to be used for rebalancing classes. 'oversample' will upsample
        the class with less samples. 'undersample' will downsample the class
        with more samples. 'weight' will generate balanced class weights.
        Default: 'oversample'

    Returns
    -------
    data : numpy array
        Rebalanced feature array of shape (n_features, n_samples)
    target : numpy array
        Corresponding class distributions. Class sizes are now evenly balanced.
    sample_weight : numpy array, optional
    """
    sample_weight = None
    if np.mean(labels) != 0.5:
        if method == "oversample":
            ros = RandomOverSampler(sampling_strategy="auto")
            data, labels = ros.fit_resample(data, labels)  # type: ignore
        elif method == "undersample":
            ros = RandomUnderSampler(sampling_strategy="auto")
            data, labels = ros.fit_resample(data, labels)  # type: ignore
        elif method == "weight":
            sample_weight = compute_sample_weight(class_weight="balanced", y=labels)
        else:
            raise ValueError(f"Method not identified. Given method was " f"{method}.")
    return data, labels, sample_weight


def features_from_csv(
    data: pd.DataFrame,
    output_dir: Union[Path, str],
    exclude_eosin: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict, dict]:
    """Get features, labels and groups from input Dataframe.

    Parameters
    ---------
        data: pd.DataFrame
            Data of nucleus measurements from .csv files exported with QuPath.
        exclude_eosin: boolean, default: True
            Set to ``False`` if eosin measurements should be used as features.

    Returns
    -------
        features: pd.DataFrame
            DataFrame of extracted features
        labels: np.ndarray
            Sample labels
        groups: np.ndarray
            Sample groups
    """
    output_dir = Path(output_dir)
    factor_classes = pd.factorize(data["Class"])
    labels = factor_classes[0]
    class_defs = dict(enumerate(factor_classes[1]))
    print("Class definitions:", class_defs)
    with open(output_dir / "class_definitions.json", "w", encoding="utf-8") as file:
        json.dump(class_defs, file)

    factor_images = pd.factorize(data["Image"])
    groups = factor_images[0]
    group_defs = dict(enumerate(factor_images[1]))
    with open(
        output_dir / "group_definitions.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(group_defs, file)

    del_cols = []
    if exclude_eosin:
        del_cols = [col for col in data.columns if "Eosin" in col]
    features = data.drop(
        columns=[
            "Image",
            "Name",
            "Class",
            "Parent",
            "ROI",
            "Centroid X µm",
            "Centroid Y µm",
        ]
        + del_cols
    )
    return features, labels, groups, class_defs, group_defs


def train_final_xgb(data: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """Train and save model using XGBoost classifier."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir()
    model = xgboost.XGBClassifier(
        booster="gbtree",
        objective="multi:softprob",
        use_label_encoder=False,
        num_class=3,
        n_estimators=250,
        max_depth=6,
        subsample=0.9,
        colsample_bylevel=0.9,
        colsample_bynode=0.9,
        colsample_bytree=0.9,
        learning_rate=0.3,
        eval_metric=["merror", "mlogloss"],
    )

    features_df, labels, groups, _, _ = features_from_csv(
        data, output_dir, exclude_eosin=True
    )
    features = features_df.values
    feature_list = features_df.columns

    # Perform validation split
    val_split = GroupShuffleSplit(n_splits=1, train_size=0.9)
    train_ind, val_ind = next(val_split.split(features, labels, groups))
    features_train, features_val = (
        features[train_ind],
        features[val_ind],
    )
    labels_train, labels_val = (
        labels[train_ind],
        labels[val_ind],
    )

    # Balance labels
    features_train, labels_train, sample_weight = balance_samples(
        data=features_train, labels=labels_train, method="oversample"
    )

    data_train = pd.DataFrame(data=features_train, columns=feature_list)
    labels_train = pd.DataFrame(data=labels_train)

    eval_set = [
        (
            pd.DataFrame(data=features_val, columns=feature_list),
            pd.DataFrame(data=labels_val, columns=["Label"]),
        )
    ]

    # Fit outer model
    model.fit(
        X=data_train,
        y=labels_train,
        eval_set=eval_set,
        sample_weight=sample_weight,
        early_stopping_rounds=50,
        verbose=False,
    )

    model.save_model(fname=Path(output_dir) / "model.json")


def cross_validate_xgb(
    data: pd.DataFrame,
    output_dir: Union[Path, str],
    cross_validator=GroupKFold(n_splits=5),
    verbose: bool = True,
) -> None:
    """Run cross-validation using XGBoost classifier."""
    output_dir = Path(output_dir)
    models_dir = output_dir / "crossval_models"
    models_dir.mkdir(exist_ok=True)
    predictions_dir = output_dir / "crossval_predictions"
    predictions_dir.mkdir(exist_ok=True)

    features_df, labels, groups, _, group_defs = features_from_csv(
        data, output_dir, exclude_eosin=True
    )
    features = features_df.values
    feature_list = features_df.columns

    current_groups = []
    accuracies = []
    balanced_accuracies = []
    expected_accuracies = []

    for fold, (train_index, test_index) in enumerate(
        cross_validator.split(features, labels, groups)
    ):
        print("Fold no.:", fold)
        features_train, features_test = (
            features[train_index],
            features[test_index],
        )
        labels_train, labels_test = labels[train_index], labels[test_index]
        groups_train = groups[train_index]
        current_group = np.array([group_defs[i] for i in np.unique(groups[test_index])])
        current_groups.append(current_group)
        print("Current test groups: ", current_group)

        # Train outer model with optimized parameters
        model = xgboost.XGBClassifier(
            booster="gbtree",
            objective="multi:softprob",
            use_label_encoder=False,
            num_class=3,
            n_estimators=250,
            max_depth=6,
            subsample=0.9,
            colsample_bylevel=0.9,
            colsample_bynode=0.9,
            colsample_bytree=0.9,
            learning_rate=0.3,
            eval_metric=["merror", "mlogloss"],
        )

        # Perform validation split
        val_split = GroupShuffleSplit(n_splits=1, train_size=0.9)
        train_ind, val_ind = next(
            val_split.split(features_train, labels_train, groups_train)
        )
        features_train, features_val = (
            features_train[train_ind],
            features_train[val_ind],
        )
        labels_train, labels_val = (
            labels_train[train_ind],
            labels_train[val_ind],
        )

        # Balance labels
        features_train, labels_train, sample_weight = balance_samples(
            features_train, labels_train, "oversample"
        )

        data_train = pd.DataFrame(data=features_train, columns=feature_list)
        labels_train = pd.DataFrame(labels_train)

        eval_set = [
            (
                pd.DataFrame(data=features_val, columns=feature_list),
                pd.DataFrame(labels_val),
            )
        ]

        # Fit outer model
        model.fit(
            X=data_train,
            y=labels_train,
            eval_set=eval_set,
            sample_weight=sample_weight,
            early_stopping_rounds=50,
            verbose=False,
        )

        # Make predictions
        labels_pred = model.predict(
            X=features_test,
        )
        acc = accuracy_score(y_true=labels_test, y_pred=labels_pred)
        bal_acc = balanced_accuracy_score(y_true=labels_test, y_pred=labels_pred)
        accuracies.append(acc)
        balanced_accuracies.append(bal_acc)
        result = model.evals_result()
        expected_acc = 1 - result["validation_0"]["merror"][-1]  # type: ignore
        expected_accuracies.append(expected_acc)

        model.save_model(fname=models_dir / ("model_" + str(fold) + ".json"))

        pd.DataFrame(
            data=list(zip(labels_test, labels_pred)),
            columns=[
                "Labels",
                "Predictions",
            ],
        ).to_csv(
            path_or_buf=predictions_dir / ("predictions_" + str(fold) + ".csv"),
            index=False,
        )

        if verbose:
            print(
                (
                    f"Expected Accuracy: {expected_acc:.2f},"
                    f" Actual Accuracy: {acc:.2f},"
                    f" Actual Balanced Accuracy: {bal_acc:.2f}"
                )
            )
    if verbose:
        print(
            (
                f"Expected MeanAccuracy: {np.mean([expected_accuracies]):.2f},"
                f" Actual Mean Accuracy: {np.mean([accuracies]):.2f},"
                " Actual Mean Balanced Accuracy:"
                f" {np.mean([balanced_accuracies]):.2f}"
            )
        )
    pd.DataFrame(
        data=zip(current_groups, expected_accuracies, accuracies, balanced_accuracies),  # type: ignore
        columns=[
            "Tested Groups",
            "Expected Accuracy",
            "Actual Accuracy",
            "Actual Balanced Accuracy",
        ],
    ).to_csv(path_or_buf=output_dir / "scores.csv", index=True)
