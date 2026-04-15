import pytest
from pathlib import Path
import gates.config as config_module
from gates.config import Config, root_dir


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Reset the global config cache before each test."""
    config_module.config_cache = None
    yield
    config_module.config_cache = None


@pytest.mark.skipif(
    not (root_dir / "config.yml").exists(),
    reason="No config.yml found — run `python gates/config.py` to create one first.",
)

class TestConfigWithExistingFile:
    """
    Tests for the Config class, that check if the existing config.yml is formatted as expected.
    It requires a config.yml file to be present, otherwise the tests are skipped.
    """
    
    def test_config_instantiates(self):
        cfg = Config()
        assert cfg is not None

    def test_minimum_keys_present(self):
        cfg = Config()
        for key in config_module.minimum_config_keys:
            assert hasattr(cfg, key), f"Config missing expected key: {key}"

    def test_derived_path_attributes_are_paths(self):
        cfg = Config()
        assert isinstance(cfg.fp_datadir, Path)
        assert isinstance(cfg.met_datadir, Path)
        assert isinstance(cfg.topog_datadir, Path)
        assert isinstance(cfg.save_models_dir, Path)
        assert isinstance(cfg.parameter_files_dir, Path)

    def test_landcover_datadir_is_path_or_none(self):
        cfg = Config()
        assert cfg.landcover_datadir is None or isinstance(cfg.landcover_datadir, Path)

    def test_root_and_package_dirs_set(self):
        cfg = Config()
        assert isinstance(cfg.root_dir, Path)
        assert isinstance(cfg.package_dir, Path)

    def test_domains_is_dict(self):
        cfg = Config()
        assert isinstance(cfg.domains, dict)

    def test_bad_files_is_list(self):
        cfg = Config()
        assert isinstance(cfg.bad_files, list)

    def test_config_is_read_only(self):
        cfg = Config()
        with pytest.raises(AttributeError, match="read-only"):
            cfg.fp_datadir = Path("/some/other/path")
