from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

# Headless backend so the script can run in terminal/HPC jobs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Ensure imports work when running from repository root or from vis/.
REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, VIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model.data.load_data import LoadBaseSatelliteData
from vis_plotting.fp_plotting import (
	compute_footprint_availability_metrics,
	plot_footprint_availability,
	plot_seasonal_footprint_histogram,
	write_footprint_availability_txt,
)
from vis_plotting.general_plotting import plot_domain, plot_multiple_data_series
from vis_plotting.met_plotting import compute_wind_speed_metrics, write_wind_speed_metrics_txt
from vis_plotting.topog_plotting import (
	plot_landuse_frequency,
	plot_majority_landuse_map,
	plot_topography,
	plot_topography_histogram,
)


REGION_DOMAIN_HINTS = {
	"BRAZIL": "SOUTHAMERICA",
	"SOUTHAMERICA": "SOUTHAMERICA",
	"SAHARA": "NORTHAFRICA",
	"INDIA": "SOUTHASIA",
}


def _safe_filename(name: str) -> str:
	"""Convert plot names into filesystem-safe filenames."""
	return (
		name.strip()
		.lower()
		.replace(" ", "_")
		.replace("/", "_")
		.replace("(", "")
		.replace(")", "")
	)


def _save_fig(fig: plt.Figure, out_dir: Path, filename_stem: str, dpi: int = 220) -> Path:
	"""Save and close a matplotlib figure."""
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{_safe_filename(filename_stem)}.png"
	fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)
	return out_path


def _country_list_from_arg(countries_arg: Optional[str]) -> Optional[list[str]]:
	"""Parse countries from JSON list or comma-separated string."""
	if countries_arg is None:
		return None

	countries_arg = countries_arg.strip()
	if not countries_arg:
		return None

	if countries_arg.startswith("["):
		parsed = json.loads(countries_arg)
		if not isinstance(parsed, list):
			raise ValueError("--countries JSON value must be a list.")
		return [str(c).strip().upper() for c in parsed if str(c).strip()]

	return [c.strip().upper() for c in countries_arg.split(",") if c.strip()]


def _print_available_regions() -> None:
	"""Print known region names and their default UM domains."""
	print("Available built-in regions (case-insensitive):")
	for region, dom in REGION_DOMAIN_HINTS.items():
		print(f" - {region} (default domain: {dom})")
	print("\nIf you need a custom region, pass both --region and --domain explicitly.")


def _print_available_countries(data: LoadBaseSatelliteData) -> None:
	"""Print countries available in the loaded region/domain country mask."""
	country_names = sorted(
		str(c).strip().upper() for c in np.asarray(data.countries.country_mask["name"]).ravel()
	)
	print(f"Available countries for region={data.region}, domain={data.domain}: {len(country_names)}")
	print("Countries:")
	for name in country_names:
		print(f" - {name}")


def _pick_default_country(data: LoadBaseSatelliteData, fallback: str = "BRAZIL") -> str:
	"""Pick a valid available country from mask names when possible."""
	names = [str(c).strip().upper() for c in np.asarray(data.countries.country_mask["name"]).ravel()]
	if fallback.upper() in names:
		return fallback.upper()
	if names:
		return names[0]
	return fallback.upper()


def _pick_release_point_indices(data: LoadBaseSatelliteData) -> tuple[int, int]:
	"""Find nearest met grid indices to the median release position."""
	rel_lat = float(np.nanmedian(data.fp_data_full["release_lat"].values))
	rel_lon = float(np.nanmedian(data.fp_data_full["release_lon"].values))

	lat_idx = int(np.abs(data.met_file["lat"].values - rel_lat).argmin())
	lon_idx = int(np.abs(data.met_file["lon"].values - rel_lon).argmin())
	return lat_idx, lon_idx


def _run_domain_plots(data: LoadBaseSatelliteData, out_dir: Path) -> list[Path]:
	"""Generate domain-boundary plots at global extent.

	Parameters
	----------
	data : LoadBaseSatelliteData
		Loaded data container with domain-aligned met/fp/topography datasets.
	out_dir : Path
		Directory where generated figures are saved.

	Returns
	-------
	list[Path]
		Paths to saved domain plot files.
	"""
	saved: list[Path] = []

	for dtype in ("met", "fp", "topo"):
		fig, _ = plot_domain(
			data,
			type=dtype,
			title=f"{dtype.upper()} Domain",
			zoom_to_data=False,
		)
		saved.append(_save_fig(fig, out_dir, f"domain_{dtype}"))

	fig, _ = plot_multiple_data_series(
		data,
		show=("met", "fp", "topo"),
		title="UM Domains: Met, Footprints, and Topography",
		zoom_to_union=False,
	)
	saved.append(_save_fig(fig, out_dir, "domain_overlay_met_fp_topo"))
	return saved


def _run_footprint_plots(data: LoadBaseSatelliteData, out_dir: Path) -> list[Path]:
	"""Generate footprint visualizations and availability summaries.

	Parameters
	----------
	data : LoadBaseSatelliteData
		Loaded data container with footprint dataset in ``fp_data_full``.
	out_dir : Path
		Directory where generated files are saved.

	Returns
	-------
	list[Path]
		Paths to saved footprint files (figures and metrics text).
	"""
	saved: list[Path] = []

	fig = plot_seasonal_footprint_histogram(data.fp_data_full, metric="count", return_fig=True)
	saved.append(_save_fig(fig, out_dir, "footprint_seasonal_count"))

	fig = plot_seasonal_footprint_histogram(data.fp_data_full, metric="mean_sum", return_fig=True)
	saved.append(_save_fig(fig, out_dir, "footprint_seasonal_mean_sum"))

	fig = plot_footprint_availability(data.fp_data_full, return_fig=True)
	saved.append(_save_fig(fig, out_dir, "footprint_availability"))

	metrics = compute_footprint_availability_metrics(data.fp_data_full)
	availability_path = out_dir / "footprint_availability_metrics.txt"
	write_footprint_availability_txt(
		metrics,
		availability_path,
		region=data.region,
		domain=data.domain,
		date=data.date,
	)
	saved.append(availability_path)
	return saved


def _run_topography_plots(
	data: LoadBaseSatelliteData,
	out_dir: Path,
	country: Optional[str],
) -> list[Path]:
	"""Generate topography and land-use plots.

	Parameters
	----------
	data : LoadBaseSatelliteData
		Loaded data container.
	out_dir : Path
		Directory where generated figures are saved.
	country : Optional[str]
		Country name to mask to. If ``None``, runs globally over the loaded domain.

	Returns
	-------
	list[Path]
		Paths to saved topography/land-use plot files.
	"""
	saved: list[Path] = []

	# Include domain name in filename when country is specified
	domain = data.domain if country else None
	location_suffix = f"{domain}_{country}" if country and domain else (country or "global")

	fig, _, _, _ = plot_topography(
		data,
		n_bins=10,
		vmin=0,
		vmax=4000,
		country=country,
		plot_country_boundaries=True,
	)
	saved.append(_save_fig(fig, out_dir, f"topography_map_{location_suffix}"))

	fig, _, _ = plot_topography_histogram(
		data,
		bins=100,
		country=country,
		log_y=True,
		density=False,
		title="Land-only Topography Histogram",
	)
	saved.append(_save_fig(fig, out_dir, f"topography_histogram_{location_suffix}"))

	fig, _, _ = plot_landuse_frequency(data, country=country, area_weighted=True)
	saved.append(_save_fig(fig, out_dir, f"landuse_frequency_{location_suffix}"))

	fig, _, _, _, _ = plot_majority_landuse_map(data, country=country, title="Majority land-use")
	saved.append(_save_fig(fig, out_dir, f"landuse_majority_map_{location_suffix}"))
	return saved


def _run_windrose_plot(data: LoadBaseSatelliteData, out_dir: Path, level: int = 1) -> list[Path]:
	"""Generate an optional wind rose at the median release location.

	Parameters
	----------
	data : LoadBaseSatelliteData
		Loaded data container with meteorology dataset.
	out_dir : Path
		Directory where generated figures are saved.
	level : int, optional
		Vertical model level to plot if ``levels`` is present, by default 1.

	Returns
	-------
	list[Path]
		Saved file paths. Returns an empty list if windrose plotting is unavailable.
	"""
	saved: list[Path] = []

	try:
		from vis_plotting.met_plotting import plot_single_windrose
	except Exception as exc:
		print(f"[warn] Skipping windrose plot because met plotting import failed: {exc}")
		return saved

	lat_idx, lon_idx = _pick_release_point_indices(data)
	met_point = data.met_file.isel(lat=lat_idx, lon=lon_idx)

	if "levels" in met_point.dims:
		met_point = met_point.sel(levels=level)

	wax = plot_single_windrose(
		met_point,
		limit=15,
		legend=True,
		title=f"Per-footprint Wind Rose - level {level}",
	)
	fig = wax.figure
	saved.append(_save_fig(fig, out_dir, f"met_windrose_level_{level}"))
	return saved


def _run_wind_speed_metrics(data: LoadBaseSatelliteData, out_dir: Path, level: int = 1) -> list[Path]:
	"""Compute wind-speed metrics and save them as a text file.

	Parameters
	----------
	data : LoadBaseSatelliteData
		Loaded data container with meteorology dataset.
	out_dir : Path
		Directory where metrics text file will be saved.
	level : int, optional
		Vertical model level used for wind statistics, by default 1.

	Returns
	-------
	list[Path]
		Single-item list containing the saved metrics file path, or empty on skip.
	"""
	saved: list[Path] = []

	try:
		metrics = compute_wind_speed_metrics(data.met_file, level=level)
	except ValueError as exc:
		print(f"[warn] Skipping metrics file: {exc}")
		return saved

	out_dir.mkdir(parents=True, exist_ok=True)
	metrics_path = out_dir / "wind_speed_metrics.txt"
	write_wind_speed_metrics_txt(
		metrics,
		metrics_path,
		region=data.region,
		domain=data.domain,
		date=data.date,
	)

	saved.append(metrics_path)
	return saved


def run_characterisation_batch(
	date: str,
	region: str,
	out_dir: Optional[Path] = None,
	domain: Optional[str] = None,
	do_align_domains: bool = True,
	countries: Optional[Iterable[str]] = None,
	include_domain: bool = True,
	include_footprint: bool = True,
	include_topography: bool = True,
	include_windrose: bool = True,
	include_metrics: bool = True,
	verbose: bool = True,
) -> list[Path]:
	"""Load one data object and run selected plot groups, saving all figures.

	Parameters
	----------
	date : str
		Date selector passed to the loader (single month/year or pattern).
	region : str
		Region name used by ``LoadBaseSatelliteData``.
	out_dir : Optional[Path], optional
		Output directory. If ``None``, a timestamped directory is created.
	domain : Optional[str], optional
		Optional UM domain override. If ``None``, inferred from region hints.
	do_align_domains : bool, optional
		If ``True``, crop datasets to common intersection before plotting.
	countries : Optional[Iterable[str]], optional
		Optional country list. First country is used for topography masking.
	include_domain : bool, optional
		Include domain-boundary maps.
	include_footprint : bool, optional
		Include footprint plots and availability summaries.
	include_topography : bool, optional
		Include topography and land-use plots.
	include_windrose : bool, optional
		Include windrose plot.
	include_metrics : bool, optional
		Include wind-speed metrics text output.
	verbose : bool, optional
		Pass-through verbosity flag for data loading.

	Returns
	-------
	list[Path]
		Paths to all saved output files.

	If out_dir is None, creates a timestamped folder including the domain name.
	"""
	countries = list(countries) if countries is not None else None
	region_upper = region.upper()

	if domain is None and region_upper not in REGION_DOMAIN_HINTS:
		raise ValueError(
			f"Unknown region '{region}'. Use one of {list(REGION_DOMAIN_HINTS.keys())} or pass --domain explicitly."
		)

	data = LoadBaseSatelliteData(
		date,
		region=region_upper,
		domain=domain,
		load_everything=True,
		verbose=verbose,
	)

	# If no out_dir provided, create one with domain included
	if out_dir is None:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		out_dir = (
			REPO_ROOT
			/ "vis"
			/ "outputs"
			/ f"characterisation_{date}_{data.domain.lower()}_{region.lower()}_{timestamp}"
		)

	if do_align_domains:
		data.align_domains(crop_to_intersection=True)

	data.get_country_masks()
	_print_available_countries(data)

	# Only apply a country mask when the user explicitly passes --countries.
	target_country = countries[0] if countries else None

	saved_files: list[Path] = []

	if include_domain:
		saved_files.extend(_run_domain_plots(data, out_dir))

	if include_footprint:
		saved_files.extend(_run_footprint_plots(data, out_dir))

	if include_topography:
		saved_files.extend(_run_topography_plots(data, out_dir, country=target_country))

	if include_windrose:
		saved_files.extend(_run_windrose_plot(data, out_dir, level=1))

	if include_metrics:
		saved_files.extend(_run_wind_speed_metrics(data, out_dir, level=1))

	return saved_files


def _build_arg_parser() -> argparse.ArgumentParser:
	"""Build CLI parser for the characterisation batch runner.

	Returns
	-------
	argparse.ArgumentParser
		Configured parser containing all command-line flags.
	"""
	parser = argparse.ArgumentParser(description="Generate characterisation plots in one run.")
	parser.add_argument(
		"--date",
		type=str,
		required=False,
		help=(
			"Date/pattern passed to loader, e.g. 201602, 2016, 2016* or 201[4-6]* "
			"for multi-year loading in one run"
		),
	)
	parser.add_argument("--region", type=str, default="BRAZIL", help="Region name, e.g. BRAZIL")
	parser.add_argument("--domain", type=str, default=None, help="Optional domain override")
	parser.add_argument("--list-regions", action="store_true", help="Print available built-in regions and exit")
	parser.add_argument(
		"--out-dir",
		type=str,
		default=None,
		help="Output folder for PNG files. Default: vis/outputs/characterisation_<date>_<region>_<timestamp>",
	)
	parser.add_argument(
		"--countries",
		type=str,
		default=None,
		help='Optional countries list, either "BRAZIL,ARGENTINA" or JSON like "[\"BRAZIL\", \"ARGENTINA\"]".',
	)
	parser.add_argument("--no-align-domains", action="store_true", help="Skip domain intersection alignment")
	parser.add_argument("--no-domain", action="store_true", help="Skip domain plots")
	parser.add_argument("--no-footprint", action="store_true", help="Skip footprint plots")
	parser.add_argument("--no-topography", action="store_true", help="Skip topography/landuse plots")
	parser.add_argument("--no-windrose", action="store_true", help="Skip windrose plot")
	parser.add_argument("--no-metrics", action="store_true", help="Skip wind speed metrics text output")
	parser.add_argument("--quiet", action="store_true", help="Reduce loader print verbosity")
	parser.add_argument("--all", action="store_true", help="Run all plot groups (default behavior)")
	return parser


def main() -> None:
	"""Parse CLI arguments, run batch characterisation, and print outputs."""
	parser = _build_arg_parser()
	args = parser.parse_args()

	if args.list_regions:
		_print_available_regions()
		return

	if not args.date:
		parser.error("--date is required unless --list-regions is used")

	countries = _country_list_from_arg(args.countries)

	out_dir = Path(args.out_dir) if args.out_dir else None

	saved = run_characterisation_batch(
		date=args.date,
		region=args.region,
		domain=args.domain,
		out_dir=out_dir,
		do_align_domains=not args.no_align_domains,
		countries=countries,
		include_domain=not args.no_domain,
		include_footprint=not args.no_footprint,
		include_topography=not args.no_topography,
		include_windrose=not args.no_windrose,
		include_metrics=not args.no_metrics,
		verbose=not args.quiet,
	)

	print("\nSaved plots:")
	for path in saved:
		print(f" - {path}")


if __name__ == "__main__":
	main()