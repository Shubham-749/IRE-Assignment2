"""Q2 (Option A): LightGBM re-ranker correctness on a small synthetic training frame."""

import random

import polars as pl

from ire_a2.reranker import build_training_frame, feature_importance, score, train_gbdt


def _synthetic_features(n_impressions: int = 30, n_candidates: int = 5) -> pl.DataFrame:
    """One feature ("signal") perfectly correlated with the label plus a noise
    feature -- a trained ranker should learn to put the label==1 row first almost
    every time.
    """
    rng = random.Random(0)
    rows = []
    for i in range(n_impressions):
        pos = rng.randrange(n_candidates)
        for j in range(n_candidates):
            label = 1 if j == pos else 0
            rows.append(
                {
                    "impression_id": f"imp{i}",
                    "article_id": f"A{j}",
                    "label": label,
                    "signal": label * 10.0 + rng.uniform(0, 0.1),
                    "noise": rng.uniform(0, 1),
                }
            )
    return pl.DataFrame(rows)


def test_build_training_frame_drops_all_zero_and_all_one_label_impressions():
    frame = pl.DataFrame(
        {
            "impression_id": ["a", "a", "b", "b", "c", "c"],
            "article_id": ["x", "y", "x", "y", "x", "y"],
            "label": [0, 0, 1, 1, 1, 0],  # a: all-0, b: all-1, c: mixed -- only c should survive
            "signal": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    kept = build_training_frame(frame)
    assert kept["impression_id"].unique().to_list() == ["c"]


def test_gbdt_ranks_the_true_click_above_distractors():
    frame = _synthetic_features()
    train = build_training_frame(frame)
    feature_cols = ["signal", "noise"]
    model = train_gbdt(train, feature_cols)

    correct = 0
    n = 0
    for imp_id in train["impression_id"].unique().to_list():
        sub = train.filter(pl.col("impression_id") == imp_id)
        scores = score(model, sub, feature_cols)
        top_label = sub["label"].to_list()[int(scores.argmax())]
        correct += top_label == 1
        n += 1
    assert correct / n > 0.9


def test_feature_importance_ranks_the_correlated_feature_highest():
    frame = _synthetic_features()
    train = build_training_frame(frame)
    feature_cols = ["signal", "noise"]
    model = train_gbdt(train, feature_cols)
    importances = feature_importance(model, feature_cols)
    assert importances[0][0] == "signal"
