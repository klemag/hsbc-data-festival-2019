import json

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GoalAdjustor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({"adjusted_goal": X.goal * X.static_usd_rate})


class CategoriesExtractor(BaseEstimator, TransformerMixin):
    def _get_slug(self, x):
        categories = json.loads(x).get("slug", "/").split("/")

        return categories

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categories = X["category"]

        return pd.DataFrame(
            {
                "gen_cat": categories.apply(lambda x: self._get_slug(x)[0]),
                "precise_cat": categories.apply(lambda x: self._get_slug(x)[1]),
            }
        )
