from pathlib import Path
import json

ROOT_DIR = Path(__file__).parents[1]
secrets_fpath = Path(ROOT_DIR, "secrets", "secrets.json")

with open(secrets_fpath) as content:
    secrets = json.load(content)

DTN_CREDENTIALS = secrets["dtn_api"]

PI_DATABASE_PATH = "\\\\CORP-PISQLAF\\Onward Energy"
