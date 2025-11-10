from pathlib import Path
import json

ROOT_DIR = Path(__file__).parents[1]
secrets_fpath = Path(ROOT_DIR, "secrets", "secrets.json")

with open(secrets_fpath) as content:
    secrets = json.load(content)  # returns dict

DTN_CREDENTIALS = secrets.get("dtn_api", None)

FRACSUN_API_CREDENTIALS = secrets.get("fracsun_api", None)

PI_DATABASE_PATH = "\\\\CORP-PISQLAF\\Onward Energy"
