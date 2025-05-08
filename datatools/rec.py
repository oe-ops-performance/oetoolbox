from pathlib import Path
import pandas as pd
from ..utils.helpers import quiet_print_function


def read_rec_file(filepath, delimiter="\t", q=True):
    """
    Reads a .rec file and returns a list of records, where each record is a list of fields.

    Parameters
    ----------
        filename : str | Path
            The full path to the .rec file.
        delimiter : str, optional
            The field delimiter. Defaults to tab ('\t').
        q : bool, optional
            Quiet parameter that suppresses printouts when True

    Returns
    -------
        list
            A list of records, or an empty list if an error occurs.
    """
    qprint = quiet_print_function(q=q)
    fpath = Path(filepath)
    if not fpath.exists():
        qprint(f"Error: file not found:\n{str(filepath)}")
        return []
    if fpath.suffix not in [".rec", ".re2"]:
        qprint("Error: unsupported file type/extension")
        return []
    qprint(f"Loading file: {fpath.name}")
    try:
        with open(filepath, "r") as file:
            records = []
            for line in file:
                line = line.strip()  # remove leading/trailing whitespace
                if line:  # skip empty lines
                    fields = line.split(delimiter)
                    records.append(fields)
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    if len(list(set([len(list_) for list_ in records]))) == 1:
        return [list_[0] for list_ in records]
    return records


def get_parameter(entry: str) -> dict:
    """Returns a dictionary with 'name' and 'value' keys for parameter"""
    param_string = entry.split("<")[1]
    param_parts = param_string.split('"')
    param_name = param_parts[1]
    param_value = param_parts[-1][1:]
    return {"name": param_name, "value": param_value}


def load_inverter_parameter_file(filepath, q=True) -> pd.DataFrame:
    """Returns a dataframe with parameters names/values and index name = device key"""
    qprint = quiet_print_function(q=q)
    records = read_rec_file(filepath, q=q)
    if not all(isinstance(entry, str) for entry in records):
        qprint("Error: unexpected file format")
        return pd.DataFrame()

    device_key_entry = list(filter(lambda e: "device key" in e, records[:10])).pop()
    if not device_key_entry:
        qprint("Error: could not find device key in file")
        return pd.DataFrame()

    full_device_key = device_key_entry.split('"')[1]

    is_parameter = lambda e: e.split("<")[1].startswith("param key=")
    parameter_list = list(map(get_parameter, filter(is_parameter, records)))
    df_parameters = pd.DataFrame(parameter_list).rename_axis(full_device_key)

    return df_parameters
