import matplotlib.pyplot as plt
import pandas as pd


def plot_target_distribution(df):
    df["y"].value_counts().plot(kind="bar")
    plt.title("Distribution of Term Deposit Subscription")
    plt.xlabel("Subscription")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_age_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["age"], bins=20, edgecolor="black")
    plt.title("Distribution of Customer Age")
    plt.xlabel("Age")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.show()


def plot_subscription_rate(df, column, title, figsize=(8, 5), horizontal=False):
    rates = pd.crosstab(
        df[column],
        df["y"],
        normalize="index"
    ).mul(100)

    if horizontal:
        rates["yes"].sort_values().plot(kind="barh", figsize=figsize)
    else:
        rates["yes"].sort_values().plot(kind="bar", figsize=figsize)

    plt.title(title)
    plt.xlabel("Subscription Rate (%)")
    plt.ylabel(column.replace("_", " ").title())
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_job_subscription(df):
    plot_subscription_rate(
        df,
        "job",
        "Term Deposit Subscription Rate by Job",
        figsize=(10, 6),
        horizontal=True
    )


def plot_education_subscription(df):
    plot_subscription_rate(
        df,
        "education",
        "Term Deposit Subscription Rate by Education Level"
    )


def plot_marital_subscription(df):
    plot_subscription_rate(
        df,
        "marital",
        "Term Deposit Subscription Rate by Marital Status"
    )


def plot_housing_subscription(df):
    plot_subscription_rate(
        df,
        "housing",
        "Term Deposit Subscription Rate by Housing Loan Status"
    )


def plot_contact_subscription(df):
    plot_subscription_rate(
        df,
        "contact",
        "Term Deposit Subscription Rate by Contact Type"
    )


def plot_month_subscription(df):
    rates = pd.crosstab(
        df["month"],
        df["y"],
        normalize="index"
    ).mul(100)

    month_order = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    rates = rates.reindex(month_order)

    plt.figure(figsize=(10, 5))
    plt.plot(
        rates.index,
        rates["yes"],
        marker="o"
    )
    plt.title("Term Deposit Subscription Rate by Month")
    plt.xlabel("Month")
    plt.ylabel("Subscription Rate (%)")
    plt.ylim(0, 60)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_poutcome_subscription(df):
    plot_subscription_rate(
        df,
        "poutcome",
        "Term Deposit Subscription Rate by Previous Campaign Outcome"
    )


def plot_correlation_heatmap(df):
    numerical_cols = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous"
    ]

    correlation_matrix = df[numerical_cols].corr().round(2)

    plt.figure(figsize=(9, 7))
    plt.imshow(
        correlation_matrix,
        cmap="coolwarm",
        aspect="auto"
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns
    )

    plt.title("Correlation Heatmap of Numerical Variables")
    plt.tight_layout()
    plt.show()


def plot_model_performance(results):
    results_plot = results.set_index("Model")

    results_plot[
        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    ].plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Performance Comparison of Realistic Prediction Models")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.legend(title="Evaluation Metric")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(results_data):
    from sklearn.metrics import roc_curve, roc_auc_score

    y_test = results_data["y_test"]
    predictions = results_data["predictions"]

    plt.figure(figsize=(10, 7))

    for name, data in predictions.items():
        fpr, tpr, _ = roc_curve(
            y_test,
            data["probabilities"]
        )

        auc = roc_auc_score(
            y_test,
            data["probabilities"]
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Realistic Prediction Models")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(results_data, model_name="Gradient Boosting"):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_test = results_data["y_test"]
    y_pred = results_data["predictions"][model_name]["predictions"]

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Subscription", "Subscription"]
    )

    disp.plot()
    plt.title(f"Confusion Matrix - Realistic {model_name}")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(results_data, model_name="Random Forest"):
    model = results_data["predictions"][model_name]["model"]
    preprocessor = results_data["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    top_features = importance.sort_values(
        "Importance",
        ascending=False
    ).head(15).sort_values(
        "Importance",
        ascending=True
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(
        f"Top 15 Feature Importances - Realistic {model_name}"
    )
    plt.tight_layout()
    plt.show()
