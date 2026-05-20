from pathlib import Path
import json

ROOT_DIR = Path(__file__).parents[1]
secrets_fpath = Path(ROOT_DIR, "secrets", "secrets.json")

with open(secrets_fpath) as content:
    secrets = json.load(content)  # returns dict

DTN_CREDENTIALS = secrets.get("dtn_api", None)

FRACSUN_API_CREDENTIALS = secrets.get("fracsun_api", None)

PI_DATABASE_PATH = "\\\\CORP-PISQLAF\\Onward Energy"

# load default settings
from .default_settings import *

# overwrite with local settings (if exist); defined in local_settings.py
LOCAL_SETTINGS_PATH = ROOT_DIR / "local_settings.py"
if LOCAL_SETTINGS_PATH.exists():
    try:
        from ..local_settings import *
    except:
        print("Warning: could not load local settings from file.")
        pass
