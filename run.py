import argparse
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from config import (DATA_DIR, MODEL_NAME, PARAMS, X_TEST, X_TRAIN, Y_TEST,
                    Y_TRAIN)
from model import build_model


def load_dataset(x_path, y_path):
    x = pd.read_csv(os.sep.join([DATA_DIR, x_path]), index_col="id")
    y = pd.read_csv(os.sep.join([DATA_DIR, y_path]))

    return x, y


def get_cross_val_score():
    X_train, y_train = load_dataset(X_TRAIN, Y_TRAIN)
    model = build_model()
    model.set_params(**PARAMS)

    cv = cross_val_score(model, X_train, y_train, cv=3)
    return {"mean": cv.mean(), "std": cv.std()}


def train_model(print_params=False):
    X_train, y_train = load_dataset(X_TRAIN, Y_TRAIN)

    model = build_model()
    model.set_params(**PARAMS)

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_NAME)


def test_model():
    X_test, y_test = load_dataset(X_TEST, Y_TEST)
    model = joblib.load(MODEL_NAME)

    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


def main():
    parser = argparse.ArgumentParser(
        description="A command line tool to manage the project"
    )
    parser.add_argument(
        "stage",
        metavar="stage",
        type=str,
        choices=["crossval", "train", "test"],
        help="Stage to run.",
    )

    stage = parser.parse_args().stage

    if stage == "crossval":
        print("Compute cross val score...")
        cv = get_cross_val_score()
        print(f"Mean: {cv['mean']}, std: {cv['std']}")

    if stage == "train":
        train_model()
        print("Model was saved")

    elif stage == "test":
        acc = test_model()
        print(f"Accuracy on test set: {acc}")


if __name__ == "__main__":
    main()
