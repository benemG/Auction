import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

def test_raw_data_dir_setting_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SETTINGS_FILE', tmp_path / 'settings.json')
    assert config.get_raw_data_dir() == config.DEFAULT_RAW_DATA_DIR
    config.set_raw_data_dir(tmp_path / 'csv')
    assert config.get_raw_data_dir() == tmp_path / 'csv'
    config.set_raw_data_dir(None)
    assert config.get_raw_data_dir() == config.DEFAULT_RAW_DATA_DIR

def test_settings_keys_do_not_overwrite_each_other(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'SETTINGS_FILE', tmp_path / 'settings.json')
    config.set_repo_spreads({'bund': -10, 'default': 0, 'nan': float('nan')})
    config.set_raw_data_dir(tmp_path / 'csv')
    assert config.get_repo_spreads() == {'bund': -10.0, 'default': 0.0}
    assert config.get_raw_data_dir() == tmp_path / 'csv'
    config.set_repo_spreads({'oat': 5})
    assert config.get_raw_data_dir() == tmp_path / 'csv'
