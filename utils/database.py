import duckdb
import pandas as pd
from pathlib import Path

from .assets import SolarSite
from .config import OPS_PERFORMANCE_DIR
from .datetime import infer_freq, upsample
from ..datatools.dchealth import CombinerMetadata
from ..dataquery.external import DTN_COLUMNS

DATABASE_FOLDER = OPS_PERFORMANCE_DIR / "python" / "scripts" / "utils" / "databases"

SUPPORTED_TABLES = [
    "dtn",
    "inverters",
    "meter",
    "met_stations",
    "combiners",
    "combiners_weighted",
    "ac_modules",
    "site",
    "pvlib",
    "pvlib_cmb",
]


def _table_name(query_group: str) -> str:
    table = query_group.lower().replace(" ", "_")
    if table not in SUPPORTED_TABLES:
        raise ValueError(f"{table = } is currently not supported.")
    return table


def _database_path(site: str) -> Path:
    site_str = site.lower().replace(" ", "-")
    db_path = DATABASE_FOLDER / f"{site_str}.db"
    if not db_path.exists():
        if not OPS_PERFORMANCE_DIR.exists():
            raise ValueError(f"Improperly configured path for 'OPS_PERFORMANCE_DIR'.")
        raise ValueError(f"No database found for '{site}'.")
    return db_path


def _format_datetime_index(df: pd.DataFrame):
    if type(df.index) is not pd.DatetimeIndex:
        raise TypeError("Dataframe must have a datetime index.")
    df = df.rename_axis("Timestamp").copy()
    if df.index.freq is None:
        freq_td = infer_freq(df.index)
        df = df.asfreq(freq_td)
    if df.index.tzinfo is not None:
        if df.index.dtype.unit != "ns":
            target_dtype = pd.DatetimeTZDtype(unit="ns", tz=df.index.tz)
            df.index = df.index.astype(target_dtype)
    return df


class SolarDatabase:

    def __init__(self, site_name: str):
        self.db_path = _database_path(site=site_name)
        self.site = SolarSite(site_name)
        self._assign_metadata()

    def _assign_metadata(self):
        with duckdb.connect(self.db_path) as con:

            # set local tz to site tz before querying (stored as utc in db)
            con.execute(f"SET TimeZone = '{self.site.timezone}';")

            # get table information
            df_tables = con.sql("SELECT * FROM duckdb_tables()").df()
            metacols = ["database_name", "table_name", "estimated_size", "column_count", "sql"]
            self.table_meta = df_tables[metacols].copy()
            self.table_names = df_tables["table_name"].to_list()

            min_tstamps, max_tstamps = [], []
            col_meta_list = []
            for table in self.table_names:
                df_ = con.sql(f"PRAGMA table_info('{table}')").df()[["name", "type"]]
                df_["table"] = table
                df_ = df_.rename(columns={"name": "column", "type": "dtype"}).set_index("table")
                col_meta_list.append(df_)

                t_min = con.execute(f"SELECT MIN(Timestamp) from {table}").fetchone()[0]
                min_tstamps.append(pd.Timestamp(t_min))
                t_max = con.execute(f"SELECT MAX(Timestamp) from {table}").fetchone()[0]
                max_tstamps.append(pd.Timestamp(t_max))

            self.column_meta = pd.concat(col_meta_list, axis=0, ignore_index=False)
            self.table_meta["min_timestamp"] = pd.Series(min_tstamps)
            self.table_meta["max_timestamp"] = pd.Series(max_tstamps)

    @property
    def table_ranges(self):
        return self.table_meta[["table_name", "min_timestamp", "max_timestamp"]].copy()

    def _validate_table_name(self, table: str):
        if table not in self.table_names:
            raise ValueError(f"No '{table}' table found in database.")

    def _group_name(self, table: str):
        if table not in self.table_names:
            return
        elif any(x in table for x in ["dtn", "pvlib"]):
            return
        return table.replace("_", " ").title().replace("Ac", "AC")

    def _table_name(self, group: str):
        table = group.lower().replace(" ", "_")
        if table not in self.table_names:
            return
        return table

    def _get_standard_freq(self, table: str):
        if table in ("dtn", "combiners", "combiners_weighted", "ac_modules", "pvlib_cmb"):
            return pd.Timedelta(hours=1)
        return pd.Timedelta(minutes=1)

    def print_table_definition(self, table: str):
        self._validate_table_name(table)
        sql_str = self.table_meta.set_index("table_name").at[table, "sql"]
        line1, col_defs = sql_str.replace(");", "").split("(")
        sql_lines = [f"{line1} ("] + [f"    {c}" for c in col_defs.split(", ")] + [");"]
        print("\n".join(sql_lines))
        return

    def get_column_names(self, table: str):
        self._validate_table_name(table)
        return self.column_meta.loc[table, "column"].to_list()

    def get_next_timestamp(self, table: str, round=True):
        self._validate_table_name(table)
        max_tstamp = self.table_meta.set_index("table_name").at[table, "max_timestamp"]
        if pd.isna(max_tstamp):
            return
        next_tstamp = max_tstamp + self._get_standard_freq(table)
        if round is False:
            return next_tstamp
        return next_tstamp.ceil("D")

    def query(self, table: str, start_date: str, end_date: str):
        self._validate_table_name(table)
        with duckdb.connect(self.db_path) as con:

            # set local tz to site tz before querying (stored as utc in db)
            con.execute(f"SET TimeZone = '{self.site.timezone}';")

            # retrieve data
            df = con.execute(f"""
                SELECT * from {table}
                WHERE Timestamp >= '{start_date}' AND Timestamp < '{end_date}'
            """).fetchdf()

        df = df.set_index("Timestamp")
        return _format_datetime_index(df)

    def _process_freq_input(self, freq, table_names):
        if freq not in (None, "1min", "1h"):
            raise ValueError("Invalid freq specified.")
        if freq is not None:
            return pd.Timedelta(freq)
        native_freq_tds = list(set(self._get_standard_freq(table) for table in table_names))
        return min(native_freq_tds)

    def query_multiple(
        self,
        start_date: str,
        end_date: str,
        table_names: list[str] = [],
        exclude_tables: list[str] = [],
        freq: str = None,
        by_table: bool = True,
    ):
        """note: if freq=None (default), uses smallest interval & resamples others."""
        if table_names == []:
            table_names = self.table_names

        valid_tables = [t for t in table_names if t in self.table_names and t not in exclude_tables]
        if not valid_tables:
            raise ValueError("No valid table names in list.")

        output_freq = self._process_freq_input(freq, valid_tables)  # for join when by_table=False

        def _processed(df):
            # resamples data when joining, or when freq is specified
            if by_table is True and freq is None:
                return df
            data_freq = infer_freq(df.index)
            if data_freq > output_freq:
                df = upsample(df, new_freq=output_freq, agg_type="interpolate")
            elif data_freq < output_freq:
                df = df.resample(output_freq).mean()
            return _format_datetime_index(df)

        data = {}
        for table in valid_tables:
            try:
                df = self.query(table, start_date, end_date)
                data.update({table: _processed(df)})
            except Exception as e:
                print(f"Warning: problem while querying {table=}; {e}")
                continue

        if len(data) == 0:
            print("No data found for specified range.")
            return

        if by_table is True:
            return data
        return pd.concat(list(data.values()), axis=1, ignore_index=False)

    def join_datasets(self, data: dict[str, pd.DataFrame], freq=None):
        valid_data = {key: df for key, df in data.items() if key in self.table_names}
        output_freq = self._process_freq_input(freq, [*valid_data])

        def _processed(df):
            data_freq = infer_freq(df.index)
            if data_freq > output_freq:
                df = upsample(df, new_freq=output_freq, agg_type="interpolate")
            elif data_freq < output_freq:
                df = df.resample(output_freq).mean()
            return _format_datetime_index(df)

        df_list = []
        for table, df_ in valid_data.items():
            try:
                df_list.append(_processed(df_))
            except Exception as e:
                print(f"Warning: problem while processing {table=}; {e}")
                continue

        if len(df_list) == 1:
            return df_list[0]
        elif len(df_list) > 1:
            return pd.concat(df_list, axis=1, ignore_index=False)
        return

    def combined_site_level_data(
        self, start_date: str, end_date: str, table_names: list[str] = [], freq: str = None
    ):
        unsupported_tables = ["combiners", "combiners_weighted", "ac_modules", "pvlib_cmb"]
        if table_names == []:
            table_names = self.table_names
        valid_tables = [
            t for t in table_names if t in self.table_names and t not in unsupported_tables
        ]
        if not valid_tables:
            raise ValueError("No valid table names in list.")
        data = self.query_multiple(
            start_date, end_date, table_names=valid_tables, freq=freq, by_table=True
        )
        df_list = []
        for key, df_ in data.items():
            if key == "meter":
                df = df_[["OE_MeterMW"]]
            elif key in ("inverters", "pvlib"):
                df = df_.sum(axis=1).div(1e3).to_frame(name=f"sum_{key}_mw")
            elif key == "met_stations":
                df = df_["Processed_POA"].to_frame(name="poa")
            elif key == "dtn":
                df = df_[["poa_global"]].rename(columns={"poa_global": "dtn_poa"})
            elif key == "site":
                df = df_
            else:
                continue
            df_list.append(df)
        dfs = pd.concat(df_list, axis=1, ignore_index=False)
        oecols = sorted([c for c in dfs.columns if c.startswith("OE")], reverse=True)
        newcols = oecols + sorted([c for c in dfs.columns if c not in oecols])
        return dfs[newcols]

    def _col_from_attpath(self, attpath: str, asset_group: str):
        # if site-level attributes, just use attribute name
        if asset_group in ("Meter", "Site"):
            return attpath.split("|")[-1]
        splitkey = asset_group  # init
        if asset_group in ["AC Modules", "Combiners", "Combiners Weighted"]:
            splitkey = "Inverters"
        *parts, att_str = attpath.split(f"{splitkey}\\")[-1].split("\\")
        return "_".join(reversed(parts + att_str.split("|")))

    def _expected_data_columns(self, table: str):
        # self._validate_table_name(table)
        if table in (
            "ac_modules",
            "inverters",
            "met_stations",
            "meter",
            "combiners",
            "combiners_weighted",
            "site",
        ):
            asset_group = table.replace("_", " ").title().replace("Ac", "AC")
            defined_attpaths = self.site.query_attributes[asset_group]
            columns = [self._col_from_attpath(p, asset_group) for p in defined_attpaths]
            if table == "met_stations":
                return [*columns, "Processed_POA"]
            return columns
        elif table == "dtn":
            return DTN_COLUMNS
        elif table == "pvlib":
            inv_names = self.site.asset_names_by_group.get("Inverters", [])
            pvl_inv_cols = [f"{inv}_Possible_Power" for inv in inv_names]
            return ["POA", *pvl_inv_cols]
        elif table == "pvlib_cmb":
            cmb_meta = CombinerMetadata(self.site.name)  # raises error if site not supported
            expected_ = []
            for inv in cmb_meta.inverter_names:
                cols_ = [f"possible_kw_{inv}"] + [
                    f"i_mp_{c}" for c in cmb_meta.rename_cols.values() if c.endswith(inv)
                ]
                expected_.extend(cols_)
            return expected_
        raise ValueError("Unexpected table name.")

    def initialize_table(self, table: str):
        if table in self.table_names:
            raise ValueError("Table already exists.")
        columns = self._expected_data_columns(table)
        table_col_str = ", ".join([f'"{col}" FLOAT' for col in columns])
        sql_command = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                Timestamp TIMESTAMPTZ,
                {table_col_str}
            );
        """
        with duckdb.connect(self.db_path) as con:
            con.execute(sql_command)
        print(f"Created new table: '{table}'")
        self._assign_metadata()
        return

    def add_new_data(self, table: str, df: pd.DataFrame, create_new: bool = False):
        if table not in SUPPORTED_TABLES:
            raise KeyError("Invalid table name.")
        elif table not in self.table_names:
            if create_new is False:
                print(f"Note: {table=} not found; set 'create_new=True'.")
                return
            self.initialize_table(table)

        expected_cols = self._expected_data_columns(table)
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing {len(missing_cols)} columns in provided df: {missing_cols}")

        next_tstamp = self.get_next_timestamp(table)
        if next_tstamp is not None:
            if next_tstamp not in df.index:
                raise ValueError(f"df must start at next timestamp in table: {next_tstamp}")
        else:
            next_tstamp = df.index[0]

        with duckdb.connect(self.db_path) as con:

            # set local tz to site tz before querying (stored as utc in db)
            con.execute(f"SET TimeZone = '{self.site.timezone}';")

            df = df[df.index >= pd.Timestamp(next_tstamp)].copy()
            if df.empty:
                print(f"No new data found in df for table '{table}'.")
                return
            df = df.reset_index(drop=False)

            # 3. Append new data to table (MAKE SURE INDEX IS SORTED AND CONTINUOUS FROM LAST TIMESTAMP IN EXISTING TABLE)
            con.execute(f"""INSERT INTO {table} SELECT * FROM df""")

        print(f">>> Added {len(df)} new rows to table: '{table}'.")
        return


def table_ranges_by_site(sitelist: list[str]):

    col_tuples = [("site", "")]
    for table_name in SUPPORTED_TABLES:
        col_tuples.extend([(table_name, "min"), (table_name, "max")])

    data_list = []
    for site in sitelist:
        try:
            db = SolarDatabase(site)
        except:
            continue
        df_ = db.table_ranges.set_index("table_name").copy()
        site_data = [site]
        for table_name in SUPPORTED_TABLES:
            if table_name not in df_.index:
                site_data.extend(["n/a", "n/a"])
                continue
            min_ = df_.at[table_name, "min_timestamp"]
            max_ = df_.at[table_name, "max_timestamp"]
            site_data.extend(
                [x.strftime("%Y-%m-%d %H:%M") if not pd.isna(x) else str(x) for x in (min_, max_)]
            )
        data_list.append(site_data)

    df = pd.DataFrame(data_list, columns=pd.MultiIndex.from_tuples(col_tuples))
    df = df.set_index(("site", "")).rename_axis(None)
    return df
