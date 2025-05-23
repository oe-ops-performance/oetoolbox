import itertools
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


def get_re2_parameter(entry: str) -> dict:
    """Returns a dictionary with 'uid' and 'value' keys for parameter"""
    entry_parts = entry.split('"')
    param_uid = entry_parts[1]
    param_value = entry_parts[3]
    return {"name": param_uid, "value": param_value}


def load_inv_param_file_re2(filepath):
    records = read_rec_file(filepath)
    metadata_end = records.index("<paramList>")
    metadata = "".join(records[:metadata_end])
    records = records[metadata_end + 2 :]
    is_parameter = lambda e: "uid=" in e
    parameter_list = list(map(get_re2_parameter, filter(is_parameter, records)))
    df_parameters = pd.DataFrame(parameter_list).rename_axis(metadata)
    return df_parameters


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


def load_and_format_parameter_file(filepath, q=True) -> pd.DataFrame:
    fpath = Path(filepath)
    df = load_inverter_parameter_file(fpath, q=q)
    df.columns = ["parameter", "value"]
    df["inverter"] = fpath.parent.name
    df["full_device_key"] = df.index.name
    df["device_key"] = df.index.name.split(":")[0]
    columns = ["inverter", "full_device_key", "device_key", "parameter", "value"]
    return df[columns].rename_axis(None)


def load_and_format_parameter_file_re2(filepath) -> pd.DataFrame:
    fpath = Path(filepath)
    df = load_inv_param_file_re2(fpath)
    df.columns = ["parameter", "value"]
    df["inverter"] = fpath.parent.name
    df["metadata"] = df.index.name
    return df.rename_axis(None)


def load_inverter_parameters(inv_param_fpaths: list, q=True) -> pd.DataFrame:
    """input: list of parameter filepaths (structured w/ parent folder = inv name)
    -> returns df w/ cols: ["inverter", "full_device_key", "device_key", "parameter", "value"]
    """
    qprint = quiet_print_function(q=q)
    if not all(fp.exists() for fp in inv_param_fpaths):
        raise ValueError("Could not find one or more filepaths. Check input.")
    df_list = []
    for fpath in inv_param_fpaths:
        try:
            df_list.append(load_and_format_parameter_file(fpath))
        except Exception as e:
            qprint(f"ERROR: {e}")
    if not df_list:
        qprint("Warning: No data loaded.")
        return
    return pd.concat(df_list, axis=0, ignore_index=True)


def load_inverter_re2_parameters(inv_param_fpaths: list) -> pd.DataFrame:
    """input: list of parameter filepaths (structured w/ parent folder = inv name)
    -> returns df w/ cols: ["inverter", "full_device_key", "device_key", "parameter", "value"]
    """
    if not all(fp.exists() for fp in inv_param_fpaths):
        raise ValueError("Could not find one or more filepaths. Check input.")
    df_list = []
    for fpath in inv_param_fpaths:
        try:
            df_list.append(load_and_format_parameter_file_re2(fpath))
        except Exception as e:
            qprint(f"ERROR: {e}")
    if not df_list:
        qprint("Warning: No data loaded.")
        return
    return pd.concat(df_list, axis=0, ignore_index=True)


def generate_summary_dataframes(df_params):
    """input: dataframe from load_inverter_parameters function"""
    df_list_1 = []  # for long-format dataframe with other metadata
    df_list_2 = []  # for wide-format dataframe with 1 row per parameter

    for param in df_params["parameter"].unique():
        dfp = df_params.loc[df_params["parameter"].eq(param)].copy()
        value_counts = dfp["value"].value_counts().to_frame()
        n_unique_values = len(value_counts)

        dfp1 = value_counts.reset_index(drop=False).copy()
        dfp1.insert(0, "parameter", param)
        dfp1["%"] = dfp1["count"].div(dfp1["count"].sum())
        dfp1["inverters"] = [
            ", ".join(dfp.loc[dfp["value"].eq(val), "inverter"].to_list()) for val in dfp1["value"]
        ]
        df_list_1.append(dfp1)

        dfp2 = value_counts.reset_index().T.reset_index(drop=True).copy()
        value_columns = [f"value_{i+1}" for i in range(n_unique_values)]
        dfp2.columns = value_columns
        for i, col in enumerate(value_columns):
            dfp2[f"%_{i+1}"] = dfp2.at[1, col] / dfp2.loc[1].sum()

        new_cols = itertools.chain.from_iterable(
            [[f"value_{i+1}", f"%_{i+1}"] for i in range(n_unique_values)]
        )
        dfp2 = dfp2[new_cols]
        dfp2.insert(0, "parameter", param)
        dfp2 = dfp2.iloc[[1], :].reset_index(drop=True).copy()
        df_list_2.append(dfp2)

    df_summary_1 = pd.concat(df_list_1, axis=0, ignore_index=True)
    df_summary_2 = pd.concat(df_list_2, axis=0, ignore_index=True)
    return df_summary_1, df_summary_2


def generate_parameter_summary_re2(df):
    """use df1 (i.e. no metadata lines)"""
    df_list = []
    for param in df["parameter"].unique():
        dfp = df.loc[df["parameter"].eq(param)].copy()
        value_counts = dfp["value"].value_counts().to_frame()
        n_unique_values = len(value_counts)

        dfp1 = value_counts.reset_index(drop=False).copy()
        dfp1.insert(0, "parameter", param)
        dfp1["%"] = dfp1["count"].div(dfp1["count"].sum())
        dfp1["inverters"] = [
            ", ".join(dfp.loc[dfp["value"].eq(val), "inverter"].to_list()) for val in dfp1["value"]
        ]
        df_list.append(dfp1)

    return pd.concat(df_list, axis=0, ignore_index=True)


def load_inverter_parameters_and_summaries(filepath_list):
    if all(Path(fp).suffix == ".rec" for fp in filepath_list):
        df_params = load_inverter_parameters(inv_param_fpaths=filepath_list)
        dfs_1, dfs_2 = generate_summary_dataframes(df_params=df_params)
        return {"parameters": df_params, "summary_1": dfs_1, "summary_2": dfs_2}
    else:
        df_params = load_inverter_re2_parameters(filepath_list)
        df = df_params.iloc[:, :-1].copy()  # drop metadata col
        df_meta = df_params[["inverter", "metadata"]].copy()
        df_meta = df_meta.drop_duplicates().reset_index(drop=True).copy()
        dfs_1 = generate_parameter_summary_re2(df)
        return {"parameters": df, "metadata": df_meta, "summary_1": dfs_1}
