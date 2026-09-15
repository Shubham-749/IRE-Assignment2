"""Q1: click-history, session, and article features for the A2 re-ranker (Q2/Q3).

Every feature depends only on `history` (already cutoff-filtered by the caller via
UserHistoryIndex.recent() -- see feature_store.py) or on a train-split-only frozen
popularity dict -- never on behaviors_val/behaviors_test or any other per-request
future state. That's the Q9 behaviour-window boundary; tests/test_a2_features.py
checks UserHistoryIndex.recent()'s cutoff directly, since that's where the boundary is
actually enforced -- this module just aggregates whatever history it's handed.
"""

import math
from pathlib import Path

import numpy as np
import polars as pl

from ire_a1.retrieval.embeddings import mean_pool

MAX_HISTORY = 20
HALF_LIFE_HOURS = 72.0
_DECAY = math.log(2) / HALF_LIFE_HOURS

FEATURE_COLUMNS = [
    "bm25_score",
    "embedding_score",
    "candidate_position",
    "hist_click_count",
    "hist_recency_weighted_count",
    "hist_category_match",
    "hist_category_match_weighted",
    "candidate_popularity",
    "freshness_hours",
    "has_freshness",
    "hist_avg_dwell_seconds",
    "session_ordinal",
    "session_click_count_so_far",
]


def _recency_weight(delta_hours: float) -> float:
    return math.exp(-_DECAY * max(delta_hours, 0.0))


def _train_popularity(behaviors_train: pl.DataFrame) -> dict:
    """article_id -> train-split click count. Frozen once from train only -- same
    "evaluation/feature-time statistic" pattern as eval/beyond_accuracy.py's
    popularity dict -- never touches val/test, so it can't leak future-split clicks.
    """
    pop = (
        behaviors_train.filter(pl.col("clicked_article_ids").is_not_null())
        .select(pl.col("clicked_article_ids").explode().alias("article_id"))
        .group_by("article_id")
        .agg(pl.len().alias("clicks"))
    )
    return dict(zip(pop["article_id"].to_list(), pop["clicks"].to_list()))


class EbnerdSessionIndex:
    """Within-session position + prior-click count, from EB-NeRD's raw
    behaviors.parquet (session_id -- not in A1's unified BEHAVIORS_SCHEMA, which drops
    it). MIND has no session_id at all, so this index is EB-NeRD only; FeatureBuilder
    leaves session_ordinal/session_click_count_so_far at 0 for MIND.

    Grouped by user once up front (same style as UserHistoryIndex) so a per-impression
    lookup is a short scan over one user's own impressions, not a full-table scan. At
    10x scale this python-level per-user list is the first thing to swap for a
    polars/join-based lookup.
    """

    def __init__(self, raw_dir: Path, scale: str):
        frames = []
        for split_dirname in ("train", "validation", "test"):
            p = raw_dir / "ebnerd" / scale / split_dirname / "behaviors.parquet"
            if not p.exists():
                continue
            raw = pl.read_parquet(p)
            has_click = (
                pl.col("article_ids_clicked").list.len() > 0
                if "article_ids_clicked" in raw.columns
                else pl.lit(False)
            )
            frames.append(
                raw.select(
                    pl.col("user_id").cast(pl.Utf8),
                    pl.col("session_id"),
                    pl.col("impression_time"),
                    has_click.alias("has_click"),
                )
            )
        combined = pl.concat(frames).sort(["user_id", "impression_time"])
        grouped = combined.group_by("user_id", maintain_order=True).agg(
            pl.col("session_id"), pl.col("impression_time"), pl.col("has_click")
        )
        self._by_user = {
            row["user_id"]: list(zip(row["session_id"], row["impression_time"], row["has_click"]))
            for row in grouped.iter_rows(named=True)
        }

    def _prior_in_session(self, user_id: str, session_id, cutoff_ts) -> list[bool]:
        return [
            has_click
            for sid, ts, has_click in self._by_user.get(user_id, ())
            if sid == session_id and ts < cutoff_ts
        ]

    def session_ordinal(self, user_id: str, session_id, cutoff_ts) -> int:
        """1-based position of this impression within its session, counting only the
        user's own earlier-timestamp impressions in the same session."""
        return len(self._prior_in_session(user_id, session_id, cutoff_ts)) + 1

    def session_click_count_so_far(self, user_id: str, session_id, cutoff_ts) -> int:
        return sum(self._prior_in_session(user_id, session_id, cutoff_ts))


def _load_ebnerd_dwell_lookup(raw_dir: Path, scale: str) -> dict:
    """(user_id, article_id, click_timestamp) -> read_time_fixed, from EB-NeRD's raw
    history.parquet -- the same source clean_ebnerd.clean_user_history() reads, so
    these keys line up exactly with UserHistoryIndex.recent()'s (article_id,
    click_timestamp) pairs. A miss (e.g. a dtype/precision edge case) just falls back
    to no dwell signal for that click rather than raising -- dwell time is an
    enrichment, not something the rest of the pipeline depends on.
    """
    out = {}
    for split_dirname in ("train", "validation", "test"):
        p = raw_dir / "ebnerd" / scale / split_dirname / "history.parquet"
        if not p.exists():
            continue
        raw = (
            pl.read_parquet(p)
            .select(
                pl.col("user_id").cast(pl.Utf8),
                pl.col("article_id_fixed"),
                pl.col("impression_time_fixed"),
                pl.col("read_time_fixed"),
            )
            .explode(["article_id_fixed", "impression_time_fixed", "read_time_fixed"])
        )
        for uid, aid, ts, rt in raw.iter_rows():
            out[(uid, str(aid), ts)] = rt
    return out


class FeatureBuilder:
    def __init__(
        self,
        dataset: str,
        articles: pl.DataFrame,
        bm25,
        emb_index,
        behaviors_train: pl.DataFrame,
        raw_dir: Path | None = None,
        scale: str | None = None,
    ):
        self.dataset = dataset
        self.bm25 = bm25
        self.emb_index = emb_index
        self._category = dict(zip(articles["article_id"].to_list(), articles["category"].to_list()))
        self._published = (
            dict(zip(articles["article_id"].to_list(), articles["published_time"].to_list()))
            if dataset == "ebnerd"
            else {}
        )
        self._popularity = _train_popularity(behaviors_train)

        # Session/dwell signal is a real EB-NeRD-only data limitation, not a modelling
        # choice -- MIND provides no session_id or per-click dwell time at all (same
        # kind of gap clean_mind.py documents for published_time/real timestamps).
        self._session_index = None
        self._dwell_lookup = {}
        if dataset == "ebnerd" and raw_dir is not None:
            self._session_index = EbnerdSessionIndex(raw_dir, scale)
            self._dwell_lookup = _load_ebnerd_dwell_lookup(raw_dir, scale)

    def build_rows(
        self,
        user_id: str,
        timestamp,
        candidate_ids: list[str],
        history: list[tuple],
        session_id=None,
    ) -> list[dict]:
        """`history`: (click_ts, article_id, title) tuples from
        UserHistoryIndex.recent(user_id, timestamp, MAX_HISTORY) -- already strictly
        before `timestamp`; this method trusts that boundary and never re-derives it.
        Returns one feature dict per candidate, in the same order as `candidate_ids`
        (so candidate_position is meaningful).
        """
        history_titles = [t for _ts, _aid, t in history if t]
        query_text = " ".join(history_titles)
        bm25_scores = self.bm25.score_candidates(query_text, candidate_ids)

        history_vecs = [v for _ts, aid, _t in history if (v := self.emb_index.get_embedding(aid)) is not None]
        query_vec = mean_pool(history_vecs)
        embedding_scores = (
            self.emb_index.score_candidates(query_vec, candidate_ids)
            if query_vec is not None
            else dict.fromkeys(candidate_ids, 0.0)
        )

        hist_click_count = min(len(history), MAX_HISTORY)
        recency_weighted_count = 0.0
        cat_count: dict = {}
        cat_weighted: dict = {}
        dwell_values = []
        for click_ts, aid, _title in history:
            delta_hours = (timestamp - click_ts).total_seconds() / 3600.0
            w = _recency_weight(delta_hours)
            recency_weighted_count += w
            cat = self._category.get(aid)
            if cat is not None:
                cat_count[cat] = cat_count.get(cat, 0) + 1
                cat_weighted[cat] = cat_weighted.get(cat, 0.0) + w
            dwell = self._dwell_lookup.get((user_id, aid, click_ts))
            if dwell is not None:
                dwell_values.append(dwell)
        hist_avg_dwell_seconds = float(np.mean(dwell_values)) if dwell_values else 0.0

        if self._session_index is not None and session_id is not None:
            session_ordinal = self._session_index.session_ordinal(user_id, session_id, timestamp)
            session_click_count_so_far = self._session_index.session_click_count_so_far(
                user_id, session_id, timestamp
            )
        else:
            session_ordinal = 0
            session_click_count_so_far = 0

        rows = []
        for pos, aid in enumerate(candidate_ids):
            cat = self._category.get(aid)
            published = self._published.get(aid)
            if published is not None:
                freshness_hours = (timestamp - published).total_seconds() / 3600.0
                has_freshness = 1
            else:
                freshness_hours = 0.0
                has_freshness = 0
            rows.append(
                {
                    "article_id": aid,
                    "bm25_score": bm25_scores.get(aid, 0.0),
                    "embedding_score": embedding_scores.get(aid, 0.0),
                    "candidate_position": pos,
                    "hist_click_count": hist_click_count,
                    "hist_recency_weighted_count": recency_weighted_count,
                    "hist_category_match": cat_count.get(cat, 0),
                    "hist_category_match_weighted": cat_weighted.get(cat, 0.0),
                    "candidate_popularity": math.log1p(self._popularity.get(aid, 0)),
                    "freshness_hours": freshness_hours,
                    "has_freshness": has_freshness,
                    "hist_avg_dwell_seconds": hist_avg_dwell_seconds,
                    "session_ordinal": session_ordinal,
                    "session_click_count_so_far": session_click_count_so_far,
                }
            )
        return rows
