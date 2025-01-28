import json
from pathlib import Path
from .oepaths import references

# get list of all JSON data files in python projects 'references' folder
json_fpaths = [fp for fp in references.glob("*.json") if "-DEN-" not in fp.name]

# function to get file reference id from filename
file_id = lambda fp: fp.stem.replace("MetaData_", "").replace("Structure_", "")

# create reference dictionary
data = {file_id(fp): json.load(open(fp)) for fp in json_fpaths}
