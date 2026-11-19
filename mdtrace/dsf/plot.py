# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot stored DSF/current spectra without re-reading the MD trajectory."""

import json
from pathlib import Path

import numpy as np


def _edges(centers):
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    middle = (centers[:-1] + centers[1:]) / 2
    return np.r_[centers[0] - (middle[0] - centers[0]), middle,
                 centers[-1] + (centers[-1] - middle[-1])]


def plot_dsf(params):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    prefix = str(params.out_files_name)
    with np.load(prefix + ".dsf.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        arrays = {name: saved[name] for name in saved.files if name != "metadata_json"}
    panels = [("sqw", "Density DSF", metadata["intensity_unit"])]
    if metadata["experiment"] == "neutron":
        panels += [("coherent_sqw", "Coherent neutron DSF", metadata["intensity_unit"]),
                   ("incoherent_sqw", "Incoherent neutron DSF", metadata["intensity_unit"])]
    if metadata["currents"]:
        panels += [("longitudinal_current", "Longitudinal particle current", metadata["current_unit"]),
                   ("transverse_current", "Transverse particle current (sum)", metadata["current_unit"])]
    energy_density = metadata.get("spectral_density_axis") == "energy_mev"
    energy_limits = (getattr(params, "dsf_plot_energy_min", None), getattr(params, "dsf_plot_energy_max", None))
    energy_axis = energy_density or any(x is not None for x in energy_limits)
    if energy_axis and getattr(params, "dsf_plot_frequency_max", None) is not None:
        raise ValueError("use energy limits instead of dsf_plot_frequency_max for an energy-axis plot")
    frequencies = arrays["energy_mev"] if energy_axis else arrays["freq_thz"]
    lower = energy_limits[0] if energy_axis and energy_limits[0] is not None else 0
    selected = frequencies >= lower
    cutoff = energy_limits[1] if energy_axis else getattr(params, "dsf_plot_frequency_max", None)
    if cutoff is not None:
        selected &= frequencies <= cutoff
    if not np.any(selected):
        raise ValueError("the requested plot range contains no saved frequency/energy bins")
    f_edges = _edges(frequencies[selected])
    spherical = metadata["q_mode"] == "spherical"
    if spherical:
        intervals = arrays["q_shell_intervals"]
        q_edges = np.unique(intervals.ravel())
        rows = np.searchsorted(q_edges, intervals[:, 0])
    else:
        q_edges = _edges(arrays["q_axis"])
    columns = min(3, len(panels))
    figure, axes = plt.subplots((len(panels) + columns - 1) // columns, columns,
                               figsize=(5.5 * columns, 4.5 * ((len(panels) + columns - 1) // columns)),
                               squeeze=False, constrained_layout=True)
    for ax, (name, title, units) in zip(axes.flat, panels):
        values = arrays[name][:, selected]
        if spherical:
            binned = np.full((len(q_edges) - 1, values.shape[1]), np.nan)
            binned[rows] = values
            values = binned
        norm = None
        positive = values[np.isfinite(values) & (values > 0)]
        if getattr(params, "dsf_plot_scale", "log") == "log" and positive.size:
            vmax = positive.max()
            vmin = max(float(np.quantile(positive, 0.05)), vmax * 1e-8)
            norm = LogNorm(vmin=min(vmin, vmax / 10), vmax=vmax)
            values = np.ma.masked_less_equal(values, 0)
        mesh = ax.pcolormesh(q_edges, f_edges, np.ma.masked_invalid(values.T),
                             shading="flat", cmap="magma", norm=norm, rasterized=True)
        figure.colorbar(mesh, ax=ax, label=units)
        ax.set(title=title, xlabel=metadata["q_axis_label"],
               ylabel="Energy transfer (meV)" if energy_axis else "Frequency (THz)")
        ax.set_ylim(bottom=lower)
        if metadata["q_mode"] == "path":
            ax.set_xticks(arrays["q_path_ticks"], metadata["q_path_labels"])
            for tick in arrays["q_path_ticks"]:
                ax.axvline(tick, color="grey", linewidth=0.5)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    figure.suptitle("Classical spectra (no one-sided intensity doubling)")
    output = Path(prefix + ".dsf.png")
    figure.savefig(output, dpi=180)
    if getattr(params, "if_show_figures", False):
        plt.show()
    plt.close(figure)
    print(f"  Saved DSF plot: {output}")
    return output
