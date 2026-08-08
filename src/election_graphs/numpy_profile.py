"""
Numpy-backed rank profiles, vendored from the ``votekit-personal`` fork.

Upstream votekit does not ship ``NumpyRankProfile`` or its helpers, so this
module carries them natively; the rest of the codebase should depend only on
released votekit. Conversions to/from ``RankProfile`` import votekit lazily.
"""

from __future__ import annotations

import csv
import io
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.errors import DataError, EmptyDataError

if TYPE_CHECKING:
    from votekit.pref_profile import RankProfile


BLANK_RANKING_SENTINEL = np.int8(-127)


@dataclass(frozen=True, slots=True)
class NumpyRankProfile:
    ballot_matrix: NDArray[np.integer]
    wt_vec: NDArray[np.floating]
    candidates: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def candidate_to_index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.candidates)}

    @property
    def total_ballot_wt(self) -> float:
        return float(self.wt_vec.sum())

    @property
    def max_ranking_length(self) -> int:
        return int(self.ballot_matrix.shape[1])


def rank_profile_to_numpy_profile(profile: RankProfile) -> NumpyRankProfile:
    from votekit.pref_profile import RankProfile

    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be of type RankProfile.")

    ranking_columns = [c for c in profile.df.columns if c.startswith("Ranking")]
    if len(ranking_columns) > len(profile.candidates):
        ranking_columns = ranking_columns[: len(profile.candidates)]

    candidate_to_index = {frozenset([name]): i for i, name in enumerate(profile.candidates)}
    candidate_to_index[frozenset()] = int(BLANK_RANKING_SENTINEL)
    candidate_to_index[frozenset(["~"])] = int(BLANK_RANKING_SENTINEL)

    cells = profile.df[ranking_columns].to_numpy() if ranking_columns else np.empty(
        (len(profile.df), 0), dtype=object
    )

    def map_cell(cell: frozenset[str]) -> int:
        try:
            return candidate_to_index[cell]
        except KeyError as exc:
            raise TypeError(f"Found invalid entry: {cell}") from exc

    ballot_matrix = np.frompyfunc(map_cell, 1, 1)(cells).astype(np.int8, copy=False)
    wt_vec = profile.df["Weight"].to_numpy(dtype=np.float64)

    return NumpyRankProfile(
        ballot_matrix=ballot_matrix,
        wt_vec=wt_vec,
        candidates=profile.candidates,
        metadata={"sentinel": int(BLANK_RANKING_SENTINEL)},
    )


def left_compact_ballot_matrix(
    ballot_matrix: NDArray[np.integer],
    sentinel: np.integer | int = np.int8(-127),
) -> NDArray[np.integer]:
    """
    Stably shift non-sentinel entries in each row left and fill the remainder with the sentinel.
    """
    if ballot_matrix.ndim != 2:
        raise ValueError("Ballot matrix must be 2-dimensional.")

    compacted = np.full_like(ballot_matrix, fill_value=sentinel)
    keep_mask = ballot_matrix != sentinel
    pos = keep_mask.cumsum(axis=1) - 1
    row_idx, col_idx = np.nonzero(keep_mask)
    compacted[row_idx, pos[row_idx, col_idx]] = ballot_matrix[row_idx, col_idx]
    return compacted


def reindex_candidate_indices(
    ballot_matrix: NDArray[np.integer],
    old_candidate_to_index: dict[str, int],
    new_candidates: Sequence[str],
    sentinel: np.integer | int = BLANK_RANKING_SENTINEL,
) -> NDArray[np.int8]:
    """
    Reindex candidate ids in a ballot matrix to match a new candidate ordering.
    """
    reindex_lut = np.full(256, int(sentinel), dtype=np.int16)
    for new_index, candidate in enumerate(new_candidates):
        reindex_lut[old_candidate_to_index[candidate] + 128] = new_index

    return reindex_lut[ballot_matrix.astype(np.int16, copy=False) + 128].astype(np.int8, copy=False)


def _normalize_removed_candidate_indices(
    profile: NumpyRankProfile,
    removed: str | int | Sequence[str | int],
) -> NDArray[np.intp]:
    removed_entries = [removed] if isinstance(removed, (str, int, np.integer)) else list(removed)

    removed_indices: list[int] = []
    for entry in removed_entries:
        if isinstance(entry, str):
            try:
                removed_indices.append(profile.candidate_to_index[entry])
            except KeyError as exc:
                raise ValueError(f"Unknown candidate to remove: {entry}") from exc
        elif isinstance(entry, (int, np.integer)):
            removed_idx = int(entry)
            if removed_idx < 0 or removed_idx >= len(profile.candidates):
                raise ValueError(f"Removal index out of range: {removed_idx}")
            removed_indices.append(removed_idx)
        else:
            raise TypeError("Removed candidates must be candidate strings or integer indices.")

    return np.asarray(sorted(set(removed_indices)), dtype=np.intp)


def _normalize_removed_candidates_with_transfer_values(
    profile: NumpyRankProfile,
    removed: str | int | Sequence[str | int],
    transfer_values: float | Sequence[float],
) -> tuple[NDArray[np.intp], dict[int, float]]:
    removed_entries = [removed] if isinstance(removed, (str, int, np.integer)) else list(removed)
    transfer_entries = (
        [float(transfer_values)]
        if isinstance(transfer_values, (int, float, np.integer, np.floating))
        else [float(value) for value in transfer_values]
    )

    if len(removed_entries) != len(transfer_entries):
        raise ValueError("transfer_values must have the same length as removed.")

    removed_indices: list[int] = []
    transfer_value_by_index: dict[int, float] = {}

    for entry, transfer_value in zip(removed_entries, transfer_entries):
        if isinstance(entry, str):
            try:
                removed_idx = profile.candidate_to_index[entry]
            except KeyError as exc:
                raise ValueError(f"Unknown candidate to remove: {entry}") from exc
        elif isinstance(entry, (int, np.integer)):
            removed_idx = int(entry)
            if removed_idx < 0 or removed_idx >= len(profile.candidates):
                raise ValueError(f"Removal index out of range: {removed_idx}")
        else:
            raise TypeError("Removed candidates must be candidate strings or integer indices.")

        if removed_idx in transfer_value_by_index:
            raise ValueError("Removed candidates must be unique when transfer values are supplied.")

        removed_indices.append(removed_idx)
        transfer_value_by_index[removed_idx] = transfer_value

    return np.asarray(sorted(removed_indices), dtype=np.intp), transfer_value_by_index


def remove_and_condense_numpy_profile(
    profile: NumpyRankProfile,
    removed: str | int | Sequence[str | int],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> NumpyRankProfile:
    """
    Remove candidate(s) from a numpy-backed rank profile and left-compact each ballot row.
    """
    if not isinstance(profile, NumpyRankProfile):
        raise TypeError("Profile must be of type NumpyRankProfile.")

    removed_indices = _normalize_removed_candidate_indices(profile, removed)
    removed_set = {profile.candidates[idx] for idx in removed_indices.tolist()}

    ballot_matrix = profile.ballot_matrix.copy()
    if removed_indices.size:
        ballot_matrix[np.isin(ballot_matrix, removed_indices)] = BLANK_RANKING_SENTINEL

    compacted_ballot_matrix = left_compact_ballot_matrix(ballot_matrix, BLANK_RANKING_SENTINEL)
    remaining_candidates = tuple(c for c in profile.candidates if c not in removed_set)

    ballot_matrix = reindex_candidate_indices(
        compacted_ballot_matrix,
        profile.candidate_to_index,
        remaining_candidates,
        BLANK_RANKING_SENTINEL,
    )

    row_keep_mask = np.ones(len(ballot_matrix), dtype=bool)
    if remove_empty_ballots:
        row_keep_mask &= ~(ballot_matrix == BLANK_RANKING_SENTINEL).all(axis=1)
    if remove_zero_weight_ballots:
        row_keep_mask &= profile.wt_vec != 0

    metadata = dict(profile.metadata)
    metadata.setdefault("sentinel", int(BLANK_RANKING_SENTINEL))

    return NumpyRankProfile(
        ballot_matrix=ballot_matrix[row_keep_mask],
        wt_vec=profile.wt_vec[row_keep_mask].copy(),
        candidates=remaining_candidates,
        metadata=metadata,
    )


def remove_and_reweigh_and_condense(
    profile: NumpyRankProfile,
    removed: str | int | Sequence[str | int],
    transfer_values: float | Sequence[float],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> NumpyRankProfile:
    """
    Remove candidate(s), reweight ballots whose first preference was removed, and left-compact.
    """
    if not isinstance(profile, NumpyRankProfile):
        raise TypeError("Profile must be of type NumpyRankProfile.")

    removed_indices, transfer_value_by_index = _normalize_removed_candidates_with_transfer_values(
        profile,
        removed,
        transfer_values,
    )
    removed_set = {profile.candidates[idx] for idx in removed_indices.tolist()}

    ballot_matrix = profile.ballot_matrix.copy()
    updated_wt_vec = profile.wt_vec.copy()

    if removed_indices.size:
        first_preferences = ballot_matrix[:, 0]
        for removed_idx, transfer_value in transfer_value_by_index.items():
            updated_wt_vec[first_preferences == removed_idx] *= transfer_value
        ballot_matrix[np.isin(ballot_matrix, removed_indices)] = BLANK_RANKING_SENTINEL

    compacted_ballot_matrix = left_compact_ballot_matrix(ballot_matrix, BLANK_RANKING_SENTINEL)
    remaining_candidates = tuple(c for c in profile.candidates if c not in removed_set)

    ballot_matrix = reindex_candidate_indices(
        compacted_ballot_matrix,
        profile.candidate_to_index,
        remaining_candidates,
        BLANK_RANKING_SENTINEL,
    )

    row_keep_mask = np.ones(len(ballot_matrix), dtype=bool)
    if remove_empty_ballots:
        row_keep_mask &= ~(ballot_matrix == BLANK_RANKING_SENTINEL).all(axis=1)
    if remove_zero_weight_ballots:
        row_keep_mask &= updated_wt_vec != 0

    metadata = dict(profile.metadata)
    metadata.setdefault("sentinel", int(BLANK_RANKING_SENTINEL))

    return NumpyRankProfile(
        ballot_matrix=ballot_matrix[row_keep_mask],
        wt_vec=updated_wt_vec[row_keep_mask],
        candidates=remaining_candidates,
        metadata=metadata,
    )


def numpy_profile_fpv(profile: NumpyRankProfile) -> dict[str, float]:
    """
    Compute first-place vote totals for a numpy-backed rank profile.
    """
    if not isinstance(profile, NumpyRankProfile):
        raise TypeError("Profile must be of type NumpyRankProfile.")

    if profile.ballot_matrix.size == 0:
        return {candidate: 0.0 for candidate in profile.candidates}

    first_preferences = profile.ballot_matrix[:, 0]
    valid_mask = first_preferences != BLANK_RANKING_SENTINEL

    if not valid_mask.any():
        return {candidate: 0.0 for candidate in profile.candidates}

    fpv_by_index = np.bincount(
        first_preferences[valid_mask],
        weights=profile.wt_vec[valid_mask],
        minlength=len(profile.candidates),
    )

    return {
        candidate: float(fpv_by_index[idx]) for idx, candidate in enumerate(profile.candidates)
    }


def numpy_profile_to_rank_profile(profile: NumpyRankProfile) -> RankProfile:
    """
    Convert a numpy-backed rank profile into a legacy RankProfile.
    """
    from votekit.pref_profile import RankProfile

    if not isinstance(profile, NumpyRankProfile):
        raise TypeError("Profile must be of type NumpyRankProfile.")

    ballot_matrix = profile.ballot_matrix
    n_rows, n_cols = ballot_matrix.shape

    lut: np.ndarray = np.empty(256, dtype=object)
    lut[:] = frozenset(["~"])
    for idx, candidate in enumerate(profile.candidates):
        lut[idx + 128] = frozenset([candidate])

    rankings = lut[ballot_matrix.astype(np.int16, copy=False) + 128]

    df = pd.DataFrame({f"Ranking_{i + 1}": rankings[:, i] for i in range(n_cols)})
    df.insert(0, "Ballot Index", np.arange(n_rows, dtype=int))
    df.set_index("Ballot Index", inplace=True)
    df["Voter Set"] = pd.Series([set() for _ in range(n_rows)], dtype=object, index=df.index)
    df["Weight"] = profile.wt_vec.astype(np.float64, copy=False)

    return RankProfile(
        max_ranking_length=profile.max_ranking_length,
        candidates=profile.candidates,
        df=df,
    )


def _normalize_frozen_fpv_indices(
    profile: NumpyRankProfile,
    freeze_fpv: Sequence[str | int] | None,
) -> NDArray[np.intp]:
    if freeze_fpv is None:
        return np.empty(0, dtype=np.intp)

    frozen_indices: list[int] = []
    for entry in freeze_fpv:
        if isinstance(entry, str):
            try:
                frozen_indices.append(profile.candidate_to_index[entry])
            except KeyError as exc:
                raise ValueError(f"Unknown candidate to freeze: {entry}") from exc
        elif isinstance(entry, (int, np.integer)):
            frozen_idx = int(entry)
            if frozen_idx < 0 or frozen_idx >= len(profile.candidates):
                raise ValueError(f"Freeze index out of range: {frozen_idx}")
            frozen_indices.append(frozen_idx)
        else:
            raise TypeError("Frozen candidates must be candidate strings or integer indices.")

    return np.asarray(sorted(set(frozen_indices)), dtype=np.intp)


def mentions_from_numpy_arrays(
    profile: NumpyRankProfile, freeze_fpv: Sequence[str | int] | None = None
) -> dict[str, float]:
    """
    Calculates total mentions for all candidates in a ``NumpyRankProfile``.

    Args:
        profile (NumpyRankProfile): Numpy-backed rank profile.
        freeze_fpv (Sequence[str | int] | None, optional): Candidates to freeze out by
            first-preference. Rows whose first entry is a frozen candidate are excluded from the
            count, and frozen candidates are omitted from the returned dictionary.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to mention totals (values).
    """
    if not isinstance(profile, NumpyRankProfile):
        raise TypeError("Profile must be of type NumpyRankProfile.")

    frozen_indices = _normalize_frozen_fpv_indices(profile, freeze_fpv)
    included_candidates = [
        candidate
        for idx, candidate in enumerate(profile.candidates)
        if idx not in set(frozen_indices.tolist())
    ]

    if profile.ballot_matrix.size == 0:
        return {c: 0.0 for c in included_candidates}

    ballot_matrix = profile.ballot_matrix
    wt_vec = profile.wt_vec

    if frozen_indices.size > 0:
        row_mask = ~np.isin(ballot_matrix[:, 0], frozen_indices)
        ballot_matrix = ballot_matrix[row_mask]
        wt_vec = wt_vec[row_mask]

    if ballot_matrix.size == 0:
        return {c: 0.0 for c in included_candidates}

    valid_mask = ballot_matrix != BLANK_RANKING_SENTINEL
    if not valid_mask.any():
        return {c: 0.0 for c in included_candidates}

    weights = np.repeat(wt_vec, valid_mask.sum(axis=1))
    mentions_by_index = np.bincount(
        ballot_matrix[valid_mask],
        weights=weights,
        minlength=len(profile.candidates),
    )

    return {
        candidate: float(mentions_by_index[idx])
        for idx, candidate in enumerate(profile.candidates)
        if idx not in set(frozen_indices.tolist())
    }


def load_numpy(
    fpath: Union[str, os.PathLike, Path],
) -> tuple[NumpyRankProfile, int, list[str], dict[str, str], str]:
    """
    Given a file path, loads cast vote record from the Scottish election csv format
    directly into a ``NumpyRankProfile``.

    Args:
        fpath (Union[str, os.PathLike, pathlib.Path]): Path to Scottish election csv file.
            Can be a url.

    Raises:
        FileNotFoundError: If fpath is invalid.
        EmptyDataError: If dataset is empty.
        DataError: If there is missing or incorrect metadata or candidate data.

    Returns:
        tuple:
            A tuple ``(NumpyRankProfile, seats, cand_list, cand_to_party, ward)``
            representing the election, the number of seats in the election, the candidate
            names, a dictionary mapping candidates to their party, and the ward.
    """

    def parse_csv_reader(reader: csv.reader) -> list[list[str]]:
        rows = [list(filter(None, row)) for row in reader]
        return [row for row in rows if row]

    fpath = str(fpath)

    if not os.path.isfile(fpath):
        with urllib.request.urlopen(fpath) as response:
            data = response.read().decode("utf-8")
        reader = csv.reader(io.StringIO(data))
        data = parse_csv_reader(reader)
    else:
        if os.path.getsize(fpath) == 0:
            raise EmptyDataError(f"CSV at {fpath} is empty.")
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = parse_csv_reader(reader)

    if len(data[0]) != 2:
        raise DataError(
            "The metadata in the first row should be number of candidates, seats."
        )

    try:
        cand_num = int(data[0][0])
        seats = int(data[0][1])
    except ValueError as exc:
        raise DataError("The first row metadata must contain integer candidate and seat counts.") from exc

    ward = data[-1][0]

    data_cand_num = len([r for r in data if "Candidate" in str(r[0])])
    if data_cand_num != cand_num:
        raise DataError(
            "Incorrect number of candidates in either first row metadata or"
            " in candidate list at end of csv file."
        )

    candidate_rows = data[len(data) - (cand_num + 1) : -1]
    for line in candidate_rows:
        if "Candidate" not in str(line[0]):
            raise DataError(
                f"The number of candidates on line 1 is {cand_num}, which does not match"
                f" the metadata."
            )

    cand_list = [line[1] for line in candidate_rows]
    cand_to_party = {line[1]: line[2] for line in candidate_rows}
    ballot_rows = data[1 : len(data) - (cand_num + 1)]

    ballot_df = pd.DataFrame(ballot_rows, dtype=object)

    if ballot_df.empty:
        ballot_matrix = np.empty((0, 0), dtype=np.int8)
        wt_vec = np.empty(0, dtype=np.float64)
    else:
        weights = pd.to_numeric(ballot_df.iloc[:, 0], errors="raise").to_numpy(dtype=np.float64)

        if ballot_df.shape[1] == 1:
            ballot_matrix = np.empty((len(ballot_df), 0), dtype=np.int8)
        else:
            ranking_df = ballot_df.iloc[:, 1:].replace("", np.nan)
            ranking_arr = ranking_df.to_numpy(dtype=np.float64, copy=True)
            valid_mask = ~np.isnan(ranking_arr)

            ranking_arr[valid_mask] -= 1
            invalid_mask = valid_mask & ((ranking_arr < 0) | (ranking_arr >= cand_num))
            if invalid_mask.any():
                bad_value = int(ranking_arr[invalid_mask][0] + 1)
                raise DataError(f"Ballot contains invalid candidate index {bad_value}.")

            ranking_arr[~valid_mask] = BLANK_RANKING_SENTINEL
            ballot_matrix = ranking_arr.astype(np.int8, copy=False)

        if ballot_matrix.size == 0:
            wt_vec = weights
        else:
            ballot_matrix, inverse = np.unique(ballot_matrix, axis=0, return_inverse=True)
            wt_vec = np.bincount(inverse, weights=weights, minlength=len(ballot_matrix)).astype(
                np.float64,
                copy=False,
            )

    profile = NumpyRankProfile(
        ballot_matrix=ballot_matrix,
        wt_vec=wt_vec,
        candidates=tuple(cand_list),
        metadata={
            "seats": seats,
            "cand_to_party": cand_to_party,
            "ward": ward,
            "source_format": "scottish_csv",
        },
    )

    return (profile, seats, cand_list, cand_to_party, ward)
