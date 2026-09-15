"""Q2 (Option A): a LightGBM LambdaRank re-ranker over Q1's engineered features.

Trains on the cached candidate-level feature table scripts/build_reranker_features.py
writes (one row per impression x candidate, with a `label` column already attached).
"""

import polars as pl


def build_training_frame(features: pl.DataFrame) -> pl.DataFrame:
    """Drop impressions with an all-0 or all-1 label -- LambdaRank needs both classes
    to learn a pairwise preference, same rule run_eval_harness._score_impression()
    already uses for AUC/nDCG -- then sort by impression_id so each impression's rows
    are contiguous, which LightGBM's `group` parameter requires.
    """
    label_sums = features.group_by("impression_id").agg(
        pl.col("label").sum().alias("n_pos"), pl.len().alias("n_total")
    )
    keep_ids = label_sums.filter((pl.col("n_pos") > 0) & (pl.col("n_pos") < pl.col("n_total")))["impression_id"]
    return features.filter(pl.col("impression_id").is_in(keep_ids.implode())).sort("impression_id")


def train_gbdt(train_frame: pl.DataFrame, feature_cols: list[str], **lgb_params):
    """LightGBM LambdaRank over `feature_cols`, grouped by impression_id. Fixed,
    modest hyperparameters (not a search) -- min_child_samples is lowered from
    LightGBM's default of 20 because demo/small-scale impressions have very few
    candidates each, and the default would refuse to split at all on this data.
    """
    import lightgbm as lgb

    X = train_frame.select(feature_cols).to_numpy()
    y = train_frame["label"].to_numpy()
    groups = train_frame.group_by("impression_id", maintain_order=True).len()["len"].to_numpy()

    params = dict(
        objective="lambdarank",
        metric="ndcg",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        min_child_samples=5,
        verbosity=-1,
    )
    params.update(lgb_params)
    model = lgb.LGBMRanker(**params)
    model.fit(X, y, group=groups)
    return model


def score(model, frame: pl.DataFrame, feature_cols: list[str]):
    return model.predict(frame.select(feature_cols).to_numpy())


def feature_importance(model, feature_cols: list[str]) -> list[tuple[str, float]]:
    importances = model.booster_.feature_importance(importance_type="gain")
    return sorted(zip(feature_cols, importances), key=lambda kv: -kv[1])
