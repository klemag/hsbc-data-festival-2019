DATA_DIR = "data"
MODEL_NAME = "model.joblib"
X_TRAIN = "X_train.zip"
Y_TRAIN = "y_train.zip"
X_TEST = "X_test.zip"
Y_TEST = "y_test.zip"


PARAMS = {
    "model__max_depth": 9,
    "model__min_samples_split": 5,
    "preprocessor__pca__n_components": 15,
}
