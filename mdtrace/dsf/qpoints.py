# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Periodic Q sampling for crystals and isotropic disordered systems.

Cells contain lattice vectors as rows, in Å. Cartesian Q includes 2*pi.
The crystal path uses MDtrace's existing commensurate-path solver. Spherical
sampling enumerates the simulation-box reciprocal lattice, not a crystal BZ.
"""

from itertools import islice, product

import numpy as np


def validate_cell(cell, name="DSF cell"):
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError(f"{name} must be a finite 3 by 3 matrix")
    if np.linalg.slogdet(cell)[0] == 0:
        raise ValueError(f"{name} must be non-singular")
    return cell


def as_qpoints(values, name):
    points = np.asarray(values, dtype=float)
    if points.size == 0 or points.size % 3 or not np.isfinite(points).all():
        raise ValueError(f"{name} requires finite groups of 3 values")
    return points.reshape(-1, 3)


def check_commensurate(points, cell, tolerance=1e-5):
    turns = points @ cell.T / (2 * np.pi)
    valid = np.all(np.abs(turns - np.rint(turns)) <= tolerance, axis=1)
    if not np.all(valid):
        index = int(np.flatnonzero(~valid)[0])
        raise ValueError(
            f"Q-point {index} is not commensurate with the simulation cell: "
            f"Q.cell/(2*pi) = {turns[index].tolist()}; each component must be "
            "an integer. Use a compatible Q point or path/spherical sampling."
        )


def _reciprocal_points(cell, q_min, q_max):
    """Yield one vector from each allowed +/- pair in bounded chunks."""
    bounds = np.floor(q_max * np.linalg.norm(cell, axis=1) / (2 * np.pi) + 1e-10).astype(int)
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    lattice = product(*(range(-int(n), int(n) + 1) for n in bounds))
    while chunk := list(islice(lattice, 32768)):
        integers = np.asarray(chunk)
        half = ((integers[:, 0] > 0)
                | ((integers[:, 0] == 0) & (integers[:, 1] > 0))
                | ((integers[:, :2] == 0).all(axis=1) & (integers[:, 2] > 0)))
        points = integers[half] @ reciprocal
        norms = np.linalg.norm(points, axis=1)
        keep = (norms >= q_min - 1e-10) & (norms <= q_max + 1e-10)
        yield points[keep], norms[keep]


def _spherical_points(cell, q_min, q_max, width, max_per_shell, seed):
    """Sample reciprocal points in +/- pairs with an optional shell-wise cap.

    Independent random priorities give a uniform sample without replacement
    within each shell. Keeping opposite vectors together avoids introducing
    a preferred sign of Q. Only a bounded chunk and the selected points are
    held in memory when a cap is used.
    """
    rng = np.random.default_rng(seed)
    selected = {}
    available = {}
    for points, norms in _reciprocal_points(cell, q_min, q_max):
        bins = np.minimum(np.floor(norms / width + 1e-10).astype(int),
                          int(np.ceil(q_max / width)) - 1)
        priorities = rng.random(len(points))
        for shell in np.unique(bins):
            mask = bins == shell
            available[shell] = available.get(shell, 0) + 2 * int(mask.sum())
            old_points, old_priorities = selected.get(shell, (np.empty((0, 3)), np.empty(0)))
            candidates = np.concatenate((old_points, points[mask]))
            scores = np.concatenate((old_priorities, priorities[mask]))
            if max_per_shell and len(candidates) > max_per_shell // 2:
                order = np.argsort(scores, kind="stable")[:max_per_shell // 2]
                candidates, scores = candidates[order], scores[order]
            selected[shell] = candidates, scores
    if not selected:
        raise ValueError("no nonzero reciprocal-box Q points fall inside dsf_q_min/dsf_q_max")
    points, groups, counts, population, intervals = [], [], [], [], []
    shell_ids = sorted(selected)
    for group, shell in enumerate(shell_ids):
        positive = selected[shell][0]
        pair = np.concatenate((positive, -positive))
        points.append(pair)
        groups.extend([group] * len(pair))
        counts.append(len(pair))
        population.append(available[shell])
        intervals.append([max(q_min, shell * width), min(q_max, (shell + 1) * width)])
    return (np.concatenate(points), np.asarray(groups), np.asarray(counts),
            np.asarray(population), np.asarray(intervals), shell_ids)


def _spherical_points_total(cell, q_min, q_max, width, max_points, seed):
    """Select a radial-balanced total budget while retaining complete +/- pairs.

    A pair receives a uniform random priority multiplied by |Q|^2. Keeping
    the smallest priorities retains all pairs below the final cutoff radius
    and progressively thins dense high-Q regions. At a fixed cutoff, the
    retention probability decreases as 1/|Q|^2, balancing the 3D radial point
    density. Selection precedes binning, so changing shell width leaves the
    sampled vectors unchanged. Only a chunk and the selected pairs are kept.
    """
    rng = np.random.default_rng(seed)
    limit = max_points // 2
    selected = np.empty((0, 3))
    priorities = np.empty(0)
    chunks, available = [], {}
    for points, norms in _reciprocal_points(cell, q_min, q_max):
        bins = np.minimum(np.floor(norms / width + 1e-10).astype(int),
                          int(np.ceil(q_max / width)) - 1)
        shells, counts = np.unique(bins, return_counts=True)
        for shell, count in zip(shells, counts):
            available[shell] = available.get(shell, 0) + 2 * int(count)
        if max_points:
            selected = np.concatenate((selected, points))
            priorities = np.r_[priorities, rng.random(len(points)) * norms**2]
            if len(selected) > limit:
                keep = np.argpartition(priorities, limit - 1)[:limit]
                selected, priorities = selected[keep], priorities[keep]
        else:
            chunks.append(points)
    if not available:
        raise ValueError("no nonzero reciprocal-box Q points fall inside dsf_q_min/dsf_q_max")
    if not max_points:
        selected = np.concatenate(chunks)
    points = np.concatenate((selected, -selected))
    bins = np.minimum(np.floor(np.linalg.norm(points, axis=1) / width + 1e-10).astype(int),
                      int(np.ceil(q_max / width)) - 1)
    order = np.argsort(bins, kind="stable")
    points, bins = points[order], bins[order]
    shells, groups, counts = np.unique(bins, return_inverse=True, return_counts=True)
    population = np.array([available[shell] for shell in shells])
    intervals = np.array([[max(q_min, shell * width), min(q_max, (shell + 1) * width)]
                          for shell in shells])
    return (points, groups, counts, population, intervals,
            sum(available.values()), sorted(available))


def build_qpoints(params, cell):
    """Return Q vectors, output-row grouping, and plotting coordinates."""
    cell = validate_cell(cell)
    mode = getattr(params, "dsf_q_mode", "explicit")
    primitive = getattr(params, "prim_unitcell", None)
    if primitive is not None:
        primitive = validate_cell(primitive, "prim_unitcell")
    ticks, labels = np.empty(0), []
    intervals = np.empty((0, 2))

    if mode == "explicit":
        points = as_qpoints(params.dsf_qpoints, "dsf_qpoints")
        if getattr(params, "dsf_q_coordinates", "fractional") == "fractional":
            if primitive is None:
                raise ValueError("fractional dsf_qpoints require prim_unitcell")
            points = 2 * np.pi * points @ np.linalg.inv(primitive).T
        axis = np.arange(len(points), dtype=float)
        axis_label = "Q-point index"
    elif mode == "path":
        from mdtrace.dsf.commensurate import BZPathHelper

        if primitive is None:
            raise ValueError("dsf_q_mode = path requires prim_unitcell")
        vertices = as_qpoints(params.dsf_q_path, "dsf_q_path")
        if len(vertices) < 2:
            raise ValueError("dsf_q_path requires at least two endpoints")
        helper = BZPathHelper(primitive, cell)
        # The existing solver recognizes nearby simple fractions. Use its
        # normalized endpoints for distances as well as generated Q vectors.
        vertices = np.array([helper._normalize_endpoint(v) for v in vertices], dtype=float)
        vertex_cart = helper.reduced_to_cartesian(vertices)
        lengths = np.linalg.norm(np.diff(vertex_cart, axis=0), axis=1)
        ticks = np.r_[0.0, np.cumsum(lengths)]
        labels = getattr(params, "dsf_q_path_labels", None) or [str(i) for i in range(len(vertices))]
        parts, distances = [], []
        for segment, (start, end) in enumerate(zip(vertices[:-1], vertices[1:])):
            qs, fractions = helper.build_commensurate_path(start, end)
            if not len(qs):
                raise ValueError(f"no commensurate Q points on dsf_q_path segment {segment}")
            if parts and np.allclose(parts[-1][-1], qs[0], rtol=0, atol=1e-10):
                qs, fractions = qs[1:], fractions[1:]
            if len(qs):
                parts.append(qs)
                distances.append(ticks[segment] + fractions * lengths[segment])
        points, axis = np.concatenate(parts), np.concatenate(distances)
        axis_label = "Path distance (Å⁻¹)"
    elif mode == "spherical":
        arguments = (cell, getattr(params, "dsf_q_min", 0.0), params.dsf_q_max,
                     getattr(params, "dsf_q_bin_width", 0.1))
        legacy_cap = getattr(params, "dsf_max_points_per_shell", None)
        seed = getattr(params, "dsf_seed", 0)
        if legacy_cap is None:
            (points, groups, counts, population, intervals,
             available_total, available_shell_ids) = _spherical_points_total(
                *arguments, getattr(params, "dsf_max_points", 5000), seed,
            )
        else:
            points, groups, counts, population, intervals, available_shell_ids = _spherical_points(
                *arguments, legacy_cap, seed,
            )
            available_total = int(population.sum())
        axis = np.bincount(groups, weights=np.linalg.norm(points, axis=1)) / counts
        axis_label = "Q magnitude (Å⁻¹)"
    else:
        raise ValueError("dsf_q_mode must be explicit, path, or spherical")

    check_commensurate(points, cell)
    if mode != "spherical":
        groups = np.arange(len(points))
        counts = population = np.ones(len(points), dtype=int)
    sampling = {
        "q_cart": points,
        "q_frac": (points @ primitive.T / (2 * np.pi)
                   if primitive is not None else np.empty((0, 3))),
        "q_box": points @ cell.T / (2 * np.pi),
        "q_group": groups, "q_counts": counts, "q_available_counts": population,
        "q_sampling_fraction": counts / population,
        "q_axis": axis, "q_axis_label": axis_label,
        "q_path_ticks": ticks, "q_path_labels": list(labels),
        "q_shell_intervals": intervals, "q_mode": mode,
    }
    if mode == "spherical":
        sampling.update(q_available_total=available_total, q_available_shells=len(available_shell_ids),
                        q_available_shell_ids=available_shell_ids,
                        q_budget_mode="total" if legacy_cap is None else "per_shell")
        sampling.update(sampling_diagnostics(points, groups, len(counts)))
    return sampling


def coverage_warning(sampling):
    """Distinguish an incomplete sample from poor coverage of all allowed vectors."""
    poor = sampling["q_coverage_poor"]
    capped = sampling["q_counts"] < sampling["q_available_counts"]
    messages = []
    if count := int(np.count_nonzero(poor & capped)):
        limit_name = ("dsf_max_points_per_shell" if sampling.get("q_budget_mode") == "per_shell"
                      else "dsf_max_points")
        messages.append(f"{count} Q shell(s) have poor directional coverage in a sampled subset; "
                        f"increase {limit_name} and reassess before judging the box")
    if count := int(np.count_nonzero(poor & ~capped)):
        messages.append(f"{count} Q shell(s) have poor directional coverage even with all allowed "
                        "box vectors; increasing the sampling cap cannot help; check cell size/shape "
                        "and shell width")
    return "; ".join(messages) + ". Inspect counts and directions in the Q sampling summary; coverage is a heuristic, not a convergence test."


def average_q_groups(values, sampling):
    """Average spectra, not amplitudes; preserve correlations in block SEMs."""
    if sampling["q_mode"] != "spherical":
        return values
    values = np.asarray(values)
    weights = np.asarray(sampling.get("q_weights", np.ones(len(values))))
    shape = (-1,) + (1,) * (values.ndim - 1)
    grouped = np.zeros((len(sampling["q_counts"]), *values.shape[1:]),
                       dtype=np.result_type(values.dtype, float))
    np.add.at(grouped, sampling["q_group"], values * weights.reshape(shape))
    totals = np.bincount(sampling["q_group"], weights=weights, minlength=len(grouped))
    return grouped / totals.reshape(shape)


def sampling_diagnostics(points, groups, n_groups, weights=None):
    """Describe sampled directions, not statistical independence or convergence.

    The second moment has eigenvalues 1/3 for isotropic coverage. It cannot
    detect all angular gaps (e.g. three orthogonal axes), so also count unique
    unoriented directions. Coverage flags are deliberately heuristic.
    """
    norms = np.linalg.norm(points, axis=1)
    weights = np.ones(len(points)) if weights is None else np.asarray(weights)
    spread, ranges, directions, eigenvalues, effective = [], [], [], [], []
    for group in range(n_groups):
        selected = groups == group
        radii, w = norms[selected], weights[selected]
        w = w / w.sum()
        spread.append(np.sqrt(np.sum(w * (radii - np.sum(w * radii))**2)))
        ranges.append([radii.min(), radii.max()])
        effective.append(1 / np.sum(w**2))
        nonzero = radii > 0
        unit = points[selected][nonzero] / radii[nonzero, None]
        if not len(unit):
            directions.append(0)
            eigenvalues.append([np.nan] * 3)
            continue
        moment = unit.T @ (unit * (w[nonzero] / w[nonzero].sum())[:, None])
        eigenvalues.append(np.linalg.eigvalsh(moment))
        first = np.argmax(np.abs(unit) > 1e-10, axis=1)
        canonical = unit * np.sign(unit[np.arange(len(unit)), first])[:, None]
        directions.append(len(np.unique(np.round(canonical, 10), axis=0)))
    eigenvalues = np.asarray(eigenvalues)
    directions = np.asarray(directions)
    return {
        "q_magnitude_std": np.asarray(spread), "q_sampled_range": np.asarray(ranges),
        "q_direction_counts": directions, "q_direction_eigenvalues": eigenvalues,
        "q_effective_count": np.asarray(effective),
        "q_coverage_poor": ((directions < 6) | (eigenvalues[:, 0] < .1)
                            | (eigenvalues[:, 2] > .6)),
    }
