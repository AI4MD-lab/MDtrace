# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot stored DSF/current spectra without re-reading the MD trajectory."""

import numpy as np
from pySED.console import print_detail

from mdtrace.dsf.results import load_spectra, plot_files, plot_has_removed_dc, plot_selection, plot_signature


_LABELS = {
    "sqw": ("Density", "S", "intensity_unit"),
    "total_current": ("Total current", r"C_{\mathrm{total}}", "current_unit"),
    "longitudinal_current": ("Longitudinal current", "C_L", "current_unit"),
    "transverse_current": ("Transverse current", "C_T", "current_unit"),
}


def _edges(centers):
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    middle = (centers[:-1] + centers[1:]) / 2
    return np.r_[centers[0] - (middle[0] - centers[0]), middle,
                 centers[-1] + (centers[-1] - middle[-1])]


def _color_limits(spectra):
    """Use the actual extrema of the visible, unmasked values in a group of maps."""
    limits = [(values.min(), values.max()) for values in spectra if values.count()]
    if limits:
        return min(low for low, _ in limits), max(high for _, high in limits)
    return None


def _draw_continuous_map(ax, values, q, freq, extent, cmap, norm):
    """Use pySED's image interpolation on regular bins and contours on physical coordinates."""
    left, right, bottom, top = extent
    dq = (right - left) / len(q)
    df = (top - bottom) / len(freq)
    if (np.allclose(q, left + (np.arange(len(q)) + 0.5) * dq)
            and np.allclose(freq, bottom + (np.arange(len(freq)) + 0.5) * df)):
        ax.imshow(values, extent=extent, origin="lower", aspect="auto", interpolation="hanning",
                  interpolation_stage="data", cmap=cmap, norm=norm)
    else:
        # Extend the outermost samples to their bin edges.
        padded = np.pad(values.astype(float, copy=False).filled(np.nan), 1, mode="edge")
        levels = np.unique(np.linspace(norm.vmin, norm.vmax, 350))
        ax.contourf(np.r_[left, q, right], np.r_[bottom, freq, top], padded,
                    levels=levels if len(levels) > 1 else 1, cmap=cmap, norm=norm, corner_mask=False)


def plot_dsf(params):
    """Write each selected map and return the path of the first figure."""
    import seaborn as sns
    from matplotlib import rc_context

    paper = {
        "axes.linewidth": 0.75, "axes.labelsize": 16,
        "xtick.major.width": 0.75, "ytick.major.width": 0.75,
        "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
    }
    style = {
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "mathtext.fontset": "dejavusans", "axes.grid": False, "axes.titlesize": 14,
    }
    with sns.axes_style("ticks"), sns.plotting_context("paper", rc=paper), rc_context(style):
        return _plot_dsf(params, sns.color_palette("RdBu_r", as_cmap=True))


def _plot_dsf(params, cmap):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    prefix = str(params.out_files_name)
    selection = plot_selection(params)
    if not any(selection.values()):
        return None
    arrays = load_spectra(prefix, for_plot=True, **selection)
    meta = arrays["metadata"]
    outputs = plot_files(params, meta)
    if "total_current" in outputs:
        arrays["total_current"] = arrays["longitudinal_current"] + arrays["transverse_current"]
    if not outputs:
        raise ValueError("No saved current spectra. Compute with dsf_currents = 1 or select dsf_plot_density = 1.")

    emin, emax = (getattr(params, key, None) for key in ("dsf_plot_energy_min", "dsf_plot_energy_max"))
    energy_axis = meta.get("spectral_density_axis") == "energy_mev" or emin is not None or emax is not None
    fmax = getattr(params, "dsf_plot_frequency_max", None)
    if energy_axis and fmax is not None:
        raise ValueError("use energy limits instead of dsf_plot_frequency_max for an energy-axis plot")
    frequencies = arrays["energy_mev"] if energy_axis else arrays["freq_thz"]
    lower = emin if energy_axis and emin is not None else 0
    upper = emax if energy_axis else fmax
    selected = frequencies >= lower
    if upper is not None:
        selected &= frequencies <= upper
    bins = np.flatnonzero(selected)
    if not len(bins):
        raise ValueError("the requested plot range contains no saved frequency/energy bins")
    f_edges = _edges(frequencies)[bins[0]:bins[-1] + 2]

    mode = meta["q_mode"]
    if mode == "spherical":
        xlabel = r"$|\mathbf{q}|$ (1/Å)"
        intervals = arrays["q_shell_intervals"]
        q = intervals.mean(axis=1)
        # Split at empty shells so interpolation never bridges missing data.
        runs = np.split(np.arange(len(q)), np.flatnonzero(intervals[1:, 0] > intervals[:-1, 1]) + 1)
        grid = meta.get("shell_grid")
        q_max = grid[1] if grid is not None else intervals[-1, 1]
        if grid is None and not meta.get("postprocessing"):
            q_max = meta.get("configuration", {}).get("dsf_q_max") or q_max
    else:
        xlabel = meta["q_axis_label"].replace("Å⁻¹", "1/Å")
        q = arrays["q_axis"]
        q_edges = _edges(q)
        intervals = np.column_stack((q_edges[:-1], q_edges[1:]))
        runs = [np.arange(len(q))]
        q_max = q_edges[-1]

    log = getattr(params, "dsf_plot_scale", "log") == "log"
    dc = arrays["freq_thz"][selected] == 0
    mask_dc = log and dc.any() and plot_has_removed_dc(params, meta)
    spectra = {}
    groups = {}
    for name in outputs:
        values = np.ma.masked_invalid(arrays[name][:, selected].T)
        if mask_dc:
            values[dc] = np.ma.masked
        spectra[name] = np.ma.log(values) if log else values
        group = _LABELS[name][2] if log else name
        groups.setdefault(group, []).append(spectra[name])
    color_limits = {key: _color_limits(group) for key, group in groups.items()}
    signature = plot_signature(params)
    show = getattr(params, "if_show_figures", False)
    figures = []
    try:
        for name, output in outputs.items():
            title, symbol, unit_key = _LABELS[name]
            figure, ax = plt.subplots(figsize=(5.5, 5), layout="constrained")
            figures.append(figure)
            limits = color_limits[unit_key if log else name]
            norm = Normalize(*limits) if limits is not None else None
            values = spectra[name]
            if mode == "explicit" and norm is not None:
                # Adjacent indices can represent unrelated Q directions.
                ax.pcolormesh(q_edges, f_edges, values, shading="flat", cmap=cmap, norm=norm)
            elif norm is not None:
                for run in runs:
                    extent = (intervals[run[0], 0], intervals[run[-1], 1], f_edges[0], f_edges[-1])
                    _draw_continuous_map(ax, values[:, run], q[run], frequencies[selected], extent, cmap, norm)
            ax.set_title(title, loc="center", pad=7)
            ax.set(xlabel=xlabel, xlim=(0, q_max), ylabel="Energy transfer (meV)" if energy_axis else r"$\omega/2\pi$ (THz)",
                   ylim=(lower, upper if upper is not None else f_edges[-1]))
            if meta["q_mode"] == "path":
                ax.set_xticks(arrays["q_path_ticks"], meta["q_path_labels"])
                for tick in arrays["q_path_ticks"]:
                    ax.axvline(tick, color="grey", linewidth=0.5)
            if norm is not None:
                bar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                                      ax=ax, pad=0.035, fraction=0.05, aspect=24)
                unit = meta[unit_key].replace("^2", "²")
                label = rf"$\ln$[${symbol}$ / ({unit})]" if log else f"${symbol}$ [{unit}]"
                bar.set_label(label, fontsize=13.5)
                bar.outline.set_visible(False)
                bar.ax.tick_params(which="both", labelsize=8, length=0, pad=0.6)
            figure.savefig(output, dpi=300, metadata={"MDtrace plot": signature})
            print_detail("Saved figure", output)
            if not show:
                plt.close(figure)
                figures.pop()
        if show:
            plt.show()
    finally:
        for figure in figures:
            plt.close(figure)
    return next(iter(outputs.values()))
