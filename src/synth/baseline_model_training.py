import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="baseline model training")

with app.setup:
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from config import load, path
    from feature_engineering import build, normalize


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Baseline model training

    Logistic regression on the same 19 selected features, split and evaluation as
    `model_training.py`, as the machine-learning baseline for the gradient-boosted tree. It is a
    linear model: it can only add up the features with one weight each, so it cannot learn
    interactions such as "a missing exit matters only when the walls are weak". If the tree beats
    it, the tree's extra complexity is earning its place.

    Logistic regression needs complete, comparably scaled inputs. Cycle positions are empty for a
    zone that never heard the tag; they are filled with -1, below every heard value (0 to 1), and
    the `read_*` flags already mark those zones. Every feature is then standardized.
    """)
    return


@app.cell
def _():
    # The 19 features and the label, and the same 80/20 split as `model_training.py` (stratified by
    # flight path, random_state 0), so both models are scored on the same test flights.
    cfg = load("configs/synth.yaml")
    selected = pd.read_parquet(path(cfg, "selected_features"))
    cols = selected.columns.drop("transition").tolist()
    flights = pd.read_parquet(path(cfg, "flights"))
    x = selected[cols]
    y = selected["transition"] == "in"
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=flights.loc[x.index, "flight_path"], random_state=0
    )
    return cfg, cols, flights, x_test, x_train, y_test, y_train


@app.cell
def _(cfg, cols, flights):
    # The hard set: straight passes the forward rule removed. `build` keeps only labelled flights,
    # so they get a placeholder label to pass through it; `removed` is joined back for the breakdown.
    _removed = flights[flights["removed"].notna()]
    hard = normalize(build(pd.read_parquet(path(cfg, "reads")), _removed.assign(transition="hard"), cfg))[
        cols
    ].join(_removed["removed"])
    return (hard,)


@app.cell
def _(x_train, y_train):
    # Fill empty positions with -1, standardize, then logistic regression with balanced class
    # weights, since `none` flights outnumber `in` about 2 to 1.
    model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-1),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    ).fit(x_train, y_train)
    return (model,)


@app.cell
def _(cols, hard, model, x_test, y_test):
    # Test scores at the 0.5 cutoff, and the share of hard-set passes called `in`.
    _p = model.predict_proba(x_test)[:, 1]
    _called_in = _p >= 0.5
    pd.Series(
        {
            "pr_auc": average_precision_score(y_test, _p),
            "roc_auc": roc_auc_score(y_test, _p),
            "precision": precision_score(y_test, _called_in),
            "recall": recall_score(y_test, _called_in),
            "f1": f1_score(y_test, _called_in),
            "hard_recall": model.predict(hard[cols]).mean(),
        }
    ).round(4).rename_axis("metric").rename("score").reset_index()
    return


@app.cell
def _(flights, model, x_test):
    # Share of test flights called `in`, by flight path. Path 0 should be near 1, the rest near 0;
    # turn-backs (1, 2) are the hard negatives.
    _path = flights.loc[x_test.index, "flight_path"]
    pd.DataFrame(
        {
            "flights": _path.value_counts().sort_index(),
            "called_in": pd.Series(model.predict(x_test), index=x_test.index).groupby(_path).mean(),
        }
    ).round(4).reset_index()
    return


@app.cell
def _(cols, hard, model):
    # Hard-set share called `in`, by why the rule removed the pass.
    pd.DataFrame(
        {
            "flights": hard["removed"].value_counts(),
            "called_in": pd.Series(model.predict(hard[cols]), index=hard.index).groupby(hard["removed"]).mean(),
        }
    ).round(4).reset_index()
    return


@app.cell
def _(model, x_train):
    # Standardized weights: how far one standard deviation of each feature moves the log-odds of
    # `in`. Positive pushes toward a crossing, negative toward `none`.
    pd.Series(model[-1].coef_[0], index=x_train.columns).sort_values(key=abs, ascending=False).round(3).rename(
        "weight"
    ).rename_axis("feature").reset_index()
    return


if __name__ == "__main__":
    app.run()
