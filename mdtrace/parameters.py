# =============================================================================
#     Copyright 2025-2026 Ting Liang and MDTRACE development team
#     This file is part of MDTRACE.
#     MDTRACE is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#     MDTRACE is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#     You should have received a copy of the GNU General Public License
#     along with MDTRACE.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""Common and DSF input parameter tables."""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class Parameter:
    """One input parameter: its default value and conversion function."""

    default: object
    read: Callable


# ==================== Value readers ====================

def _one(values):
    if len(values) != 1:
        raise ValueError("requires exactly 1 value")
    return values[0]


def read_text(values):
    return _one(values).strip("'\"")


def read_lower_text(values):
    return read_text(values).lower()


def read_int(values):
    return int(_one(values))


def read_float(values):
    return float(_one(values))


def read_bool(values):
    value = _one(values)
    if value not in {"0", "1"}:
        raise ValueError("must be 0 or 1")
    return value == "1"


def read_optional_float(values):
    value = _one(values)
    return None if value.lower() == "none" else float(value)


def read_text_list(values):
    return [value.strip("'\"") for value in values]


def read_matrix3(values):
    if len(values) != 9:
        raise ValueError("requires 9 values")
    return np.array(values, dtype=float).reshape(3, 3)


def _read_number(value):
    """Read a decimal number or a fraction such as ``1/2``."""

    try:
        return float(value)
    except ValueError:
        try:
            return float(Fraction(value))
        except ZeroDivisionError as error:
            raise ValueError("fraction denominator must not be zero") from error


def read_q_path(values):
    return np.array([_read_number(value) for value in values])


# ==================== Common parameters ====================

common_params = {
    # Control
    "action": Parameter("thinking", read_lower_text),
    "method": Parameter("dsf", read_lower_text),
    "backend": Parameter("numpy", read_lower_text),

    # Trajectory and output
    "trajectory_file": Parameter("dump.xyz", read_text),
    "out_files_name": Parameter("mdtrace", read_text),
    "lammps_unit": Parameter("metal", read_lower_text),
    "trajectory_read_mode": Parameter("cache", read_lower_text),
    "trajectory_prefetch": Parameter(True, read_bool),
    "netcdf_compression_level": Parameter(1, read_int),
    "netcdf_batch_size": Parameter(64, read_int),

    # Trajectory sampling
    "time_step": Parameter(0.0, read_float),
    "output_data_stride": Parameter(0, read_int),

    # Computation
    "num_blocks": Parameter(5, read_int),
    "max_cores": Parameter(4, read_int),

    # Reference cell for fractional DSF Q coordinates
    "prim_unitcell": Parameter(None, read_matrix3),
}


# ==================== DSF parameters ====================

dsf_params = {
    "experiment": Parameter("neutron", read_lower_text),
    "atom_types": Parameter(None, read_text_list),
    "dsf_qpoints": Parameter(None, read_q_path),
    "dsf_q_mode": Parameter("explicit", read_lower_text),
    "dsf_q_coordinates": Parameter("fractional", read_lower_text),
    "dsf_q_path": Parameter(None, read_q_path),
    "dsf_q_path_labels": Parameter(None, read_text_list),
    "dsf_q_min": Parameter(0.0, read_float),
    "dsf_q_max": Parameter(None, read_optional_float),
    "dsf_q_bin_width": Parameter(0.1, read_float),
    "dsf_max_points_per_shell": Parameter(100, read_int),
    "dsf_seed": Parameter(0, read_int),
    "dsf_cell": Parameter(None, read_matrix3),
    "dsf_currents": Parameter(False, read_bool),
    "dsf_save_q_resolved": Parameter(False, read_bool),
    "dsf_partials": Parameter(False, read_bool),
    "dsf_normalization": Parameter("density", read_lower_text),
    "dsf_window": Parameter("none", read_lower_text),
    "dsf_subtract_mean": Parameter(False, read_bool),
    "dsf_plot_frequency_max": Parameter(None, read_optional_float),
    "dsf_plot_energy_min": Parameter(None, read_optional_float),
    "dsf_plot_energy_max": Parameter(None, read_optional_float),
    "dsf_plot_scale": Parameter("log", read_lower_text),
    "if_show_figures": Parameter(False, read_bool),
}


# ==================== EELS parameters ====================

# EELS-specific parameters will be added with the calculation method.
eels_params = {}

supported_backends = {"numpy"}


# ==================== Validation ====================

def validate_common(params):
    if params.action not in {"thinking", "compute", "plot"}:
        raise ValueError("action must be thinking, compute, or plot; SED fitting is provided by pySED")
    if params.backend not in supported_backends:
        supported = ", ".join(sorted(supported_backends))
        raise ValueError(
            f"backend '{params.backend}' is not available; "
            f"supported backends: {supported}"
        )
    if params.lammps_unit not in {"metal", "real"}:
        raise ValueError("lammps_unit must be 'metal' or 'real'")
    if params.trajectory_read_mode not in {"cache", "direct"}:
        raise ValueError(
            "trajectory_read_mode must be 'cache' or 'direct'"
        )

    needs_trajectory = (
        params.method == "dsf"
        and params.action in {"thinking", "compute"}
    )
    if needs_trajectory and params.time_step <= 0:
        raise ValueError("time_step must be positive for computation")
    if needs_trajectory and params.output_data_stride <= 0:
        raise ValueError(
            "output_data_stride must be positive for computation"
        )
    if params.time_step < 0:
        raise ValueError("time_step must not be negative")
    if params.output_data_stride < 0:
        raise ValueError("output_data_stride must not be negative")
    if params.num_blocks < 1:
        raise ValueError("num_blocks must be positive")
    if params.max_cores < 1:
        raise ValueError("max_cores must be positive")
    if not 0 <= params.netcdf_compression_level <= 9:
        raise ValueError("netcdf_compression_level must be between 0 and 9")
    if params.netcdf_batch_size < 1:
        raise ValueError("netcdf_batch_size must be positive")


def validate_dsf(params, *, compute=False):
    from mdtrace.dsf.qpoints import as_qpoints, validate_cell

    value = lambda name: getattr(params, name, dsf_params[name].default)
    if compute or getattr(params, "action", "compute") in {"thinking", "compute"}:
        if getattr(params, "backend", "numpy") != "numpy":
            raise ValueError("DSF supports backend = numpy only")
        if value("experiment") not in {"neutron", "number"}:
            raise ValueError("DSF supports experiment = neutron or number only")
        if getattr(params, "trajectory_read_mode", "cache") != "cache":
            raise ValueError("DSF requires trajectory_read_mode = cache")
        if value("experiment") == "neutron" and value("atom_types") is None:
            raise ValueError("atom_types is required for neutron DSF computation")
        mode = value("dsf_q_mode")
        if mode not in {"explicit", "path", "spherical"}:
            raise ValueError("dsf_q_mode must be explicit, path, or spherical")
        if value("dsf_q_coordinates") not in {"fractional", "cartesian"}:
            raise ValueError("dsf_q_coordinates must be fractional or cartesian")
        primitive = getattr(params, "prim_unitcell", None)
        if primitive is not None:
            validate_cell(primitive, "prim_unitcell")
        if value("dsf_cell") is not None:
            validate_cell(value("dsf_cell"), "dsf_cell")
        if mode == "explicit":
            as_qpoints(value("dsf_qpoints"), "dsf_qpoints")
            if value("dsf_q_coordinates") == "fractional" and primitive is None:
                raise ValueError("fractional dsf_qpoints require prim_unitcell")
        elif mode == "path":
            if value("dsf_q_coordinates") != "fractional":
                raise ValueError("dsf_q_path uses fractional coordinates; set dsf_q_coordinates = fractional")
            if primitive is None:
                raise ValueError("dsf_q_mode = path requires prim_unitcell")
            points = as_qpoints(value("dsf_q_path"), "dsf_q_path")
            if len(points) < 2:
                raise ValueError("dsf_q_path requires at least two endpoints")
            labels = value("dsf_q_path_labels")
            if labels is not None and len(labels) != len(points):
                raise ValueError("dsf_q_path_labels must contain one label per endpoint")
        else:
            low, high, width = value("dsf_q_min"), value("dsf_q_max"), value("dsf_q_bin_width")
            if high is None or not np.isfinite([low, high, width]).all() or not (0 <= low < high and width > 0):
                raise ValueError("spherical sampling requires 0 <= dsf_q_min < dsf_q_max and positive dsf_q_bin_width")
            cap = value("dsf_max_points_per_shell")
            if cap < 0 or int(cap) != cap or cap % 2:
                raise ValueError("dsf_max_points_per_shell must be 0 (all) or a positive even integer")
            seed = value("dsf_seed")
            if seed < 0 or int(seed) != seed:
                raise ValueError("dsf_seed must be a nonnegative integer")
        if mode != "explicit" and value("dsf_qpoints") is not None:
            raise ValueError("dsf_qpoints is only used with dsf_q_mode = explicit")
        if mode != "path" and value("dsf_q_path") is not None:
            raise ValueError("dsf_q_path is only used with dsf_q_mode = path")
        if value("dsf_normalization") not in {"density", "legacy"}:
            raise ValueError("dsf_normalization must be density or legacy")
        if value("dsf_window") not in {"none", "hann"}:
            raise ValueError("dsf_window must be none or hann")
    if value("dsf_plot_scale") not in {"linear", "log"}:
        raise ValueError("dsf_plot_scale must be linear or log")
    cutoff = value("dsf_plot_frequency_max")
    if cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0):
        raise ValueError("dsf_plot_frequency_max must be finite and positive")
    emin, emax = value("dsf_plot_energy_min"), value("dsf_plot_energy_max")
    if any(x is not None and not np.isfinite(x) for x in (emin, emax)):
        raise ValueError("DSF plot energy limits must be finite")
    if emin is not None and emax is not None and emin >= emax:
        raise ValueError("dsf_plot_energy_min must be less than dsf_plot_energy_max")


def validate_eels(params):
    """EELS currently has no method-specific parameters."""


# Adding a method requires its parameter table and validator.
methods = {
    "dsf": {
        "parameters": dsf_params,
        "validate": validate_dsf,
    },
    "eels": {
        "parameters": eels_params,
        "validate": validate_eels,
    },
}


__all__ = [
    "common_params",
    "methods",
    "validate_common",
]
