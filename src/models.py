import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocessing import prepare_data


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train a model and calculate the five evaluation metrics."""
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Predictions": y_pred,
        "Probabilities": y_prob
    }


def create_models():
    """Create the four models used in the coursework."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }


def run_models(df, remove_duration=True):
    """
    Train all four models.

    remove_duration=True reproduces the realistic pre-contact experiment.
    remove_duration=False reproduces the benchmark experiment.
    """
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        preprocessor
    ) = prepare_data(df, remove_duration=remove_duration)

    models = create_models()
    results = []
    predictions = {}

    for name, model in models.items():
        metrics = evaluate_model(
            model,
            X_train_processed,
            X_test_processed,
            y_train,
            y_test
        )

        results.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "ROC-AUC": metrics["ROC-AUC"]
        })

        predictions[name] = {
            "model": model,
            "predictions": metrics["Predictions"],
            "probabilities": metrics["Probabilities"]
        }

    results_df = pd.DataFrame(results).round(4)

    return {
        "results": results_df,
        "predictions": predictions,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed
    }


def get_feature_importance(model, preprocessor, top_n=15):
    """Return the top feature importances for a tree-based model."""
    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    return importance.sort_values(
        "Importance",
        ascending=False
    ).head(top_n)
