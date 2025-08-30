import tempfile
import yaml

from pipeline.utils import ConfigManager, DatabaseManager


def test_config_get_nested_key():
    sample_config = {"level1": {"level2": {"value": 42}}}
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(sample_config, tmp)
        tmp.flush()
        cm = ConfigManager(config_path=tmp.name)
    assert cm.get("level1.level2.value") == 42


def test_database_manager_offline():
    # Provide bogus MongoDB URI to ensure connection fails and fallback is used
    cfg_dict = {
        "mongodb": {
            "uri": "mongodb://invalid_host:27017",
            "database": "testdb",
            "collection": "results"
        }
    }
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg_dict, tmp)
        tmp.flush()
        cm = ConfigManager(config_path=tmp.name)
    dbm = DatabaseManager(cm)
    # Since uri is invalid, collection should be None
    assert dbm.collection is None
