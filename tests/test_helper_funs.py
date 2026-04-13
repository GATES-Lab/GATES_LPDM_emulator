from unittest.mock import patch

from gates.data.load_data import LoadBaseSatelliteData, LoadSquareSatelliteData


def make_square_satellite_obj(
    fp_ds,
    met_ds,
    topog_ds,
    landcover_ds,
    *,
    year=2016,
    month="01",
    size=8,
    test_fp_datadir=None,
    test_met_datadir=None,
    test_topog_path=None,
    test_landcover_path=None,
    load_everything=True,
    lazy_load=True,
    load_square_kwargs=None,
):
    """Return a ``LoadSquareSatelliteData`` from either real sample files or patched datasets.

    Args:
        fp_ds: Footprint dataset used in synthetic/mock mode.
        met_ds: Meteorology dataset used in synthetic/mock mode.
        topog_ds: Topography dataset used in synthetic/mock mode.
        landcover_ds: Landcover dataset used in synthetic/mock mode.
        year: Year passed to ``LoadSquareSatelliteData``.
        month: Month passed to ``LoadSquareSatelliteData``.
        size: Square cutout size around each release point.
        test_fp_datadir: Optional real footprint sample-file path.
        test_met_datadir: Optional real meteorology sample-file path.
        test_topog_path: Optional real topography sample-file path.
        test_landcover_path: Optional real landcover sample-file path.
        load_everything: Forwarded to ``LoadSquareSatelliteData``.
        lazy_load: Forwarded to ``LoadSquareSatelliteData``.
        load_square_kwargs: Optional extra kwargs forwarded to ``LoadSquareSatelliteData``.

    Returns:
        A configured ``LoadSquareSatelliteData`` instance.
    """
    if load_square_kwargs is None:
        load_square_kwargs = {}

    paths_available = all(
        p is not None
        for p in [
            test_fp_datadir,
            test_met_datadir,
            test_topog_path,
            test_landcover_path,
        ]
    )

    if paths_available:
        return LoadSquareSatelliteData(
            year=year,
            month=month,
            size=size,
            fp_datadir=test_fp_datadir[:-9],
            met_args={"met_datadir": test_met_datadir[:-9]},
            topog_args={
                "topog_path": test_topog_path,
                "landcover_path": test_landcover_path,
            },
            load_everything=load_everything,
            lazy_load=lazy_load,
            **load_square_kwargs,
        )

    def _mock_load_fps(self_, fp_datadir):
        self_.fp_data_full = fp_ds
        self_._subsample_frequency(**self_.subsample_parameters)

    def _mock_get_met(self_, met_datadir, lazy_load=True, met_time_chunk=None, parallel=False):
        met_file = met_ds.drop_duplicates(dim=["lat", "lon", "time"])
        if "model_level_number" in met_file.dims:
            met_file = met_file.rename({"model_level_number": "levels"})
        return met_file

    def _mock_load_topog(self_, topog_path="default", landcover_path="default"):
        return self_._interp_topog(topog_ds), self_._interp_landcover(landcover_ds)

    with (
        patch.object(LoadBaseSatelliteData, "_load_footprints", _mock_load_fps),
        patch.object(LoadBaseSatelliteData, "_get_meteorology_file", _mock_get_met),
        patch.object(LoadBaseSatelliteData, "load_topog", _mock_load_topog),
    ):
        return LoadSquareSatelliteData(
            year=year,
            month=month,
            size=size,
            load_everything=load_everything,
            lazy_load=lazy_load,
            **load_square_kwargs,
        )
