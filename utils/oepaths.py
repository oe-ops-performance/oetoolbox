import datetime as dt
from pathlib import Path


# define functions for getting date created & optional string formatting
def date_created(filepath, as_string=False):
    created_datetime = dt.datetime.fromtimestamp(Path(filepath).stat().st_ctime)
    if as_string is True:
        return created_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return created_datetime


# get Shared folder that is linked to SharePoint/OneDrive
if "Onward Energy" not in Path.cwd().parts:
    onward_fp = Path(Path.home(), "Onward Energy")
else:
    idx = Path.cwd().parts.index("Onward Energy")
    onward_fp = Path(*Path.cwd().parts[: idx + 1])
shared_fp = max(
    (fp.stat().st_mtime, fp) for fp in onward_fp.iterdir() if fp.name.startswith("Shared")
)[1]
intra_fp = shared_fp if "Intralinks" in shared_fp.name else Path(shared_fp, "Intralinks")
internal_documents = Path(intra_fp, "Internal Documents")

# frequently-referenced Operations directories
operations = Path(internal_documents, "Operations")
solar = Path(operations, "Solar")
flashreports = Path(operations, "FlashReports")
python_projects = Path(operations, "PerformanceToolkit", "Python Projects")
development = Path(python_projects, "Development")
references = Path(python_projects, "References")
released = Path(python_projects, "Released")

# utility meter generation directories
commercial = Path(internal_documents, "Commercial")
standard_meter_gen = Path(commercial, "Standardized Meter Generation")
varsity_metergen = Path(standard_meter_gen, "Varsity Generation DataBase", "Monthly Generation")

UTILITY_METER_DIR = Path("\\\\corp-cdaas\\Meter_Data_Historian")

# frequently-referenced files
meter_generation_historian = Path(
    UTILITY_METER_DIR, "Master_Version", "Meter_Generation_Historian.xlsm"
)
kpi_tracker = Path(solar, "z_Monthly KPI Tracker.xlsx")
kpi_tracker_rev1 = Path(solar, "z_Monthly KPI Tracker_Rev1.xlsx")
kpi_notes_file = Path(flashreports, "_kpis_and_notes_.xlsx")


# function to generate flashreport paths
def frpath(year, month, ext=None, site=None):
    yyyymm = f"{year}{month:02d}"
    path_ = Path(flashreports, yyyymm)
    if isinstance(ext, str):
        if ext.capitalize() in ["Solar", "Wind"]:
            path_ = Path(path_, ext.capitalize())
            if isinstance(site, str):
                path_ = Path(path_, site)
    return path_


# function to generate (or validate) unique filepath
def validated_savepath(fpath):
    n_, stem_ = 1, fpath.stem
    if stem_[-1] == ")":
        split_index = stem_.rfind("(")
        stem_ = stem_[:split_index]
    while fpath.exists():
        fpath = Path(fpath.parent, f"{stem_}({n_}){fpath.suffix}")
        n_ += 1
    return fpath


# function to sort filepath list by date created
def sorted_filepaths(filepath_list):
    """sorts list of pathlib.Path filepaths by date created (most recent first)"""
    sort_kwargs = dict(key=lambda fp: fp.stat().st_ctime, reverse=True)
    return list(sorted(filepath_list, **sort_kwargs))


# function to get most recent file from filepath list
def latest_file(filepath_list):
    """returns most recently created filepath from list"""
    if not filepath_list:
        return None
    return sorted_filepaths(filepath_list)[0]


def get_contents(folder):
    if not Path(folder).is_dir() or not Path(folder).exists():
        raise ValueError("Provided path is not a valid directory.")
    output = dict(files=[], folders=[], other=[])
    for fp in sorted_filepaths(list(Path(folder).glob("*"))):
        if fp.is_dir():
            output["folders"].append(fp)
        elif fp.is_file():
            output["files"].append(fp)
        else:
            output["other"].append(fp)
    return output
