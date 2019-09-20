from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from transformers import CategoriesExtractor, GoalAdjustor


def get_preprocessor():
    cat_processor = Pipeline(
        [
            ("transformer", CategoriesExtractor()),
            ("one_hot", OneHotEncoder(sparse=False, handle_unknown="ignore")),
        ]
    )

    column_processor = ColumnTransformer(
        [
            ("goal", GoalAdjustor(), ["goal", "static_usd_rate"]),
            ("categories", cat_processor, ["category"]),
            ("disable_communication", "passthrough", ["disable_communication"]),
        ]
    )

    preprocessor = Pipeline([("column_processor", column_processor), ("pca", PCA())])
    return preprocessor


def build_model():
    preprocessor = get_preprocessor()

    model = Pipeline(
        [("preprocessor", preprocessor), ("model", DecisionTreeClassifier())]
    )

    return model
