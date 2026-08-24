"""Q5: rank_and_group() must never reorder impressions -- Codabench's grader compares
predictions to ground truth positionally, line by line, in the ground truth's original
order. A real submission was rejected once already because sorting by a string
impression_id column reordered it lexicographically ("1","10","100",... instead of
"1","2","3",...); this pins that regression down.
"""

import polars as pl

from ire_a1.eval.submission import rank_and_group


def test_output_order_matches_input_order_not_lexicographic():
    # ids deliberately include a case ("10" appearing right after "1") that would sort
    # before "2" lexicographically but must stay in original position order.
    exploded = pl.DataFrame(
        {
            "raw_impression_id": ["1", "1", "1", "2", "2", "10", "10", "10", "10", "3"],
            "pos": [0, 1, 2, 0, 1, 0, 1, 2, 3, 0],
            "score": [3.0, 1.0, 2.0, 5.0, 4.0, 9.0, 8.0, 7.0, 6.0, 0.5],
        }
    )
    result = rank_and_group(exploded, id_col="raw_impression_id")
    assert result["raw_impression_id"].to_list() == ["1", "2", "10", "3"]


def test_rank_order_is_a_valid_permutation_per_impression():
    exploded = pl.DataFrame(
        {
            "raw_impression_id": ["1", "1", "1", "2", "2"],
            "pos": [0, 1, 2, 0, 1],
            "score": [3.0, 1.0, 2.0, 5.0, 4.0],
        }
    )
    result = rank_and_group(exploded, id_col="raw_impression_id")
    for ranks in result["rank_order"].to_list():
        assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_rank_1_goes_to_highest_score():
    exploded = pl.DataFrame(
        {
            "raw_impression_id": ["1", "1", "1"],
            "pos": [0, 1, 2],
            "score": [3.0, 1.0, 2.0],  # candidate at pos=0 has the highest score
        }
    )
    result = rank_and_group(exploded, id_col="raw_impression_id")
    rank_order = result["rank_order"].to_list()[0]
    assert rank_order[0] == 1  # pos=0 (highest score) is ranked 1st
    assert rank_order.index(1) == 0
