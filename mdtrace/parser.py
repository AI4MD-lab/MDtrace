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

"""Read an MDtrace ``key = value`` input file."""

import shlex
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from mdtrace.parameters import common_params, methods, validate_common


def _read_entries(input_file):
    """Return ``(line_number, key, values)`` entries from an input file."""

    path = Path(input_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    entries = []
    first_occurrence = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            if "=" not in clean:
                raise ValueError(
                    f"{path}:{line_number}: expected 'key = value'"
                )

            key, raw_values = clean.split("=", 1)
            key = key.strip()
            try:
                values = shlex.split(
                    raw_values,
                    comments=True,
                    posix=False,
                )
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: {error}"
                ) from error
            if not key or not values:
                raise ValueError(
                    f"{path}:{line_number}: parameter or value is missing"
                )
            if key in first_occurrence:
                raise ValueError(
                    f"{path}:{line_number}: parameter '{key}' is repeated; "
                    f"first defined on line {first_occurrence[key]}"
                )
            first_occurrence[key] = line_number
            entries.append((line_number, key, values))
    return entries


def _select_method(entries):
    """Read the method before choosing its parameter table.

    Defaults to ``"dsf"`` when no ``method = ...`` line is present,
    matching the ``method`` parameter default in ``common_params``.
    """

    method = common_params["method"].default
    for _, key, values in entries:
        if key == "method":
            method = values[0].strip("'\"").lower()

    if method == "sed":
        raise ValueError("SED has been removed from MDtrace; use pySED (pysed) for SED calculation, plotting, and fitting")
    if method not in methods:
        supported = ", ".join(methods)
        raise ValueError(
            f"unsupported method '{method}'; supported methods: {supported}"
        )
    return method


def read_input(input_file="input.in"):
    """Parse an input file into a simple parameter namespace."""

    input_path = Path(input_file).expanduser().resolve()
    entries = _read_entries(input_path)
    keys = {key for _, key, _ in entries}
    if {"density_weighting", "experiment"} <= keys:
        raise ValueError(f"{input_path}: specify density_weighting or legacy experiment, not both")
    if {"dsf_max_points", "dsf_max_points_per_shell"} <= keys:
        raise ValueError(
            f"{input_path}: specify dsf_max_points or the legacy "
            "dsf_max_points_per_shell, not both"
        )
    method = _select_method(entries)
    method_config = methods[method]
    repeated_names = set(common_params) & set(method_config["parameters"])
    if repeated_names:
        names = ", ".join(sorted(repeated_names))
        raise RuntimeError(
            "parameters are defined as both common and method-specific: "
            f"{names}"
        )
    parameter_table = {
        **common_params,
        **method_config["parameters"],
    }
    defaults = {
        key: deepcopy(parameter.default)
        for key, parameter in parameter_table.items()
    }
    params = SimpleNamespace(input_file=str(input_path), **defaults)

    for line_number, key, values in entries:
        legacy_weighting = method == "dsf" and key == "experiment"
        name = "density_weighting" if legacy_weighting else key
        parameter = parameter_table.get(name)
        if parameter is None:
            raise ValueError(
                f"{input_path}:{line_number}: parameter '{key}' is not "
                f"valid for method '{method}'"
            )

        try:
            value = parameter.read(values)
            if legacy_weighting:
                if value not in {"number", "neutron"}:
                    raise ValueError("legacy experiment must be number or neutron")
                value = "equal" if value == "number" else "neutron"
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{input_path}:{line_number}: invalid '{key}': {error}"
            ) from error
        setattr(params, name, value)

    for key in ("trajectory_file",):
        if not hasattr(params, key):
            continue
        path = Path(getattr(params, key)).expanduser()
        if not path.is_absolute():
            path = input_path.parent / path
        setattr(params, key, str(path.resolve()))

    try:
        validate_common(params)
        method_config["validate"](params)
    except ValueError as error:
        raise ValueError(f"{input_path}: {error}") from error

    return params


__all__ = ["read_input"]
