import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(file_path):
    """Load the Bank Marketing dataset."""
    return pd.read_csv(file_path, sep=";")


def prepare_data(df, remove_duration=False):
    """
    Prepare the dataset for machine learning.

    remove_duration=False reproduces the benchmark setup.
    remove_duration=True creates the realistic pre-contact setup.
    """
    data = df.copy()

    X = data.drop(columns=["y"])
    y = data["y"].map({"no": 0, "yes": 1})

    if remove_duration and "duration" in X.columns:
        X = X.drop(columns=["duration"])

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        preprocessor
    )
