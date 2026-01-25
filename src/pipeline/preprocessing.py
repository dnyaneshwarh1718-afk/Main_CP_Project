import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def build_preprocessor(x: pd.DataFrame):
    """
    returns: preprocessor (ColumnTransformer)
    feature_columns (list)
    """

    # split column type

    num_cols = x.select_dtypes(include = ["int64", "float64"]).columns.tolist()
    cat_cols = x.select_dtypes(include = ["object"]).columns.tolist()

    # Numeric Pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",num_pipeline, num_cols),
            ("cat",cat_pipeline, cat_cols),
        ]
    )

    return preprocessor, num_cols, cat_cols
