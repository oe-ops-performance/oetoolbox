from pathlib import Path

CACHE = False
CACHE_DIR = Path("C:\\LOCAL_CACHE")

USER_ONWARD_PATH = Path.home() / "Onward Energy"
INTRALINKS_PATH = USER_ONWARD_PATH / "Shared - Intralinks"
OPS_LEGACY_PATH = INTRALINKS_PATH / "Internal Documents" / "Operations"

COMMERCIAL_DIR = USER_ONWARD_PATH / "Commercial CO-INT-Open - Documents"

USER_ONEDRIVE_PATH = Path.home() / "OneDrive - Onward Energy"

OPS_PERFORMANCE_DIR = USER_ONEDRIVE_PATH / "OPS - Performance Engineering - Documents"

# for weekly solar performance report spreadsheet
SOLAR_OPERATIONS_DIR = USER_ONEDRIVE_PATH / "Solar Operations - Site Assets"

SOLAR_COMMERCIAL_DIR = USER_ONEDRIVE_PATH / "Power Markets & Origination Team - Solar"
