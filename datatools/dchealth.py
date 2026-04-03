from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from .backfill import backfill_meteo_data
from .pvlib import pvlib_dataframe_to_dict, run_pvlib_combiner_level
from .qcutils import run_auto_qc
from ..reporting.query_attributes import get_solar_attributes
from ..utils import oemeta, oepaths
from ..utils.helpers import quiet_print_function
from ..utils.pi import PIDataset
from ..utils.solar import SolarDataset


@lru_cache(maxsize=128)
def load_design_db():
    fpath = oepaths.solar.joinpath("Data", "Project_Design_Database.xlsx")
    df = pd.read_excel(fpath, engine="calamine")
    df = df.dropna(axis=1, how="all").set_index("Site")

    def fmt_cols(string):
        string = string.lower()
        for x, y in [(" ", "_"), ("(", ""), (")", "")]:
            string = string.replace(x, y)
        return string

    df.columns = df.columns.map(fmt_cols)
    return df


SORTKEY = lambda x: ["".join(filter(str.isalpha, x)), int("".join(filter(str.isdigit, x)))]

CMB_SITES = [*oemeta.data["combiner_attributes"]]


class CombinerMetadata:
    def __init__(self, site: str):
        atts_by_inv = oemeta.data["combiner_attributes"].get(site, None)
        attribute_paths = get_solar_attributes(site=site).get("Combiners", None)
        if atts_by_inv is None or attribute_paths is None:
            raise ValueError("No attribute paths found for combiners.")

        self.site = site
        self.attribute_paths = attribute_paths
        self.atts_by_inv = atts_by_inv
        self.cmb_att_names = list(
            sorted(set(a for atts in self.atts_by_inv.values() for a in atts), key=SORTKEY)
        )
        self.inverter_names = list(atts_by_inv.keys())
        self.attpaths_by_inv = {
            inv: [p for p in self.attribute_paths if f"{inv}|" in p] for inv in self.inverter_names
        }
        self._get_filtered_design_database()
        self._get_dc_weights()
        self._generate_combiner_id_mapping()

    def _get_filtered_design_database(self) -> pd.DataFrame:
        """Initialization function - assigns self.ddb & self.attribute_paths.
        -> can also update/overwrite self.attpaths_by_inv & self.atts_by_inv
        """
        df_ddb = load_design_db()
        df = df_ddb.loc[self.site].copy()
        df = df.dropna(how="all", axis=1).reset_index(drop=True)
        inv_id_pi_sorted = list(sorted(df["inverter_id_pi"].unique()))
        if inv_id_pi_sorted != self.inverter_names:
            raise ValueError("ERROR: inverter name mismatch")

        # validate and remove extra rows if exist in design db
        changes = False  # init; used to overwrite existing class attributes
        attpaths_by_inv_2 = {}
        for inv, cmb_attpath_list in self.attpaths_by_inv.items():
            # get cmb ids from design db
            dfi = df.loc[df["inverter_id_pi"].eq(inv)].copy()
            if not any(c in dfi.columns for c in ["combiner_id_pi", "combiner_id_drawings"]):
                raise ValueError("no combiner id column found.")  # shouldn't happen..

            # compare PI attribute paths with combiner ids in design database (ddb)

            # case 1 - more attribute paths than defined combiner ids in ddb
            if dfi.shape[0] < len(cmb_attpath_list):
                # update attpath list to reflect combiner ids in ddb
                # this happens for GA4 - certain cmb atts marked 'Excluded' in PI
                changes = True
                if "combiner_id_pi" in df.columns:
                    # when this id exists, can use to match up with attribute paths
                    new_attpaths = []
                    for attpath in cmb_attpath_list:
                        if any(i in attpath for i in dfi["combiner_id_pi"].values):
                            new_attpaths.append(attpath)
                    # make sure there were matches for all ids
                    if len(new_attpaths) != dfi.shape[0]:
                        raise ValueError(f"{len(new_attpaths) = }, {dfi.shape[0] = }; mismatch.")
                else:
                    # remove from end of list (sequential matching assumption)
                    n_new = dfi.shape[0]
                    new_attpaths = cmb_attpath_list[:n_new]

                n_removed = len(cmb_attpath_list) - len(new_attpaths)
                print(f"Removed {n_removed} attpaths for {inv=} to match ids in design db")
                attpaths_by_inv_2[inv] = new_attpaths
                continue

            # case 2 - more combiner ids in design database than matching attributes in pi
            if dfi.shape[0] > len(cmb_attpath_list):
                # remove from design db dataframe & re-create dfi
                idx_to_remove = dfi.iloc[len(cmb_attpath_list) :].index
                df = df[~df.index.isin(idx_to_remove)].reset_index(drop=True)
                n_removed = len(idx_to_remove)
                print(f"Removed {n_removed} row(s) from ddb to match atts in pi for {inv=}")

                # update dfi for final validation check
                dfi = df.loc[df["inverter_id_pi"].eq(inv)].copy()

            if dfi.shape[0] != len(cmb_attpath_list):
                raise ValueError(f"?? mystery ?? - {dfi.shape[0] = }, {len(cmb_attpath_list)}")

            # build these in parallel; used to overwrite class attributes if changes gets set to True
            attpaths_by_inv_2[inv] = cmb_attpath_list

        # assign design database dataframe to self.ddb
        self.ddb = df
        if changes is True:
            # overwrite attribute_paths, attpaths and atts
            self.attpaths_by_inv = attpaths_by_inv_2
            self.atts_by_inv = {
                inv: [p.split("|")[-1] for p in attpaths]
                for inv, attpaths in attpaths_by_inv_2.items()
            }

        # assign list of all combiner attribute paths
        self.attribute_paths = [p for attpaths in self.attpaths_by_inv.values() for p in attpaths]

    def _generate_combiner_id_mapping(self):
        """Creates associated dictionaries for self.id_mapping & self.rename_cols."""
        id_mapping = {}
        rename_cols = {}
        for inv, cmb_att_list in self.atts_by_inv.items():
            df = self.ddb[self.ddb["inverter_id_pi"].eq(inv)].copy()
            cmb_id_list = df["combiner_id_drawings"].to_list()

            if "combiner_id_pi" in self.ddb.columns:
                df = df.set_index("combiner_id_drawings")
                # loop through ids in design db and find matching attribute names (applies to GA4)
                inv_cmb_map = {}
                for cmb_id in cmb_id_list:  # for GA4, the Q id
                    match_id = df.at[cmb_id, "combiner_id_pi"]  # for GA4, the DCCT id
                    matching_atts = [a for a in cmb_att_list if match_id in a]
                    if len(matching_atts) != 1:
                        print(f"NOTE: no matching attributes for {inv=}, {match_id=}")
                    cmb_att = matching_atts[0]
                    inv_cmb_map.update({cmb_id: cmb_att})
                    rename_cols.update({f"{cmb_att}_{inv}": f"{cmb_id}_{inv}"})
                id_mapping[inv] = inv_cmb_map
            else:
                # assumes a sequential 1:1 mapping
                id_mapping[inv] = {id_: att_ for id_, att_ in zip(cmb_id_list, cmb_att_list)}
                rename_cols.update(
                    {
                        f"{cmb_att}_{inv}": f"{cmb_id}_{inv}"
                        for cmb_att, cmb_id in zip(cmb_att_list, cmb_id_list)
                    }
                )
        self.id_mapping = id_mapping
        self.rename_cols = rename_cols

    def _get_dc_weights(self):
        """Initialization function; generates and assigns self.dc_weights"""
        df = self.ddb.rename(
            columns={
                "inverter_id_pi": "inv_id",
                "combiner_id_drawings": "cmb_id",
                "total_kwdc": "total_kw_dc",
            }
        )
        df["dc_weight"] = 0.0  # init
        df["dc_weight_adj"] = 0.0  # init
        for inv in df["inv_id"].unique():
            cond = df["inv_id"].eq(inv)
            n_cmb = df[cond].shape[0]
            df.loc[cond, "dc_weight"] = df["total_kw_dc"] / df[cond]["total_kw_dc"].mean()
            df.loc[cond, "dc_weight_adj"] = df["dc_weight"].div(n_cmb)

        self.dc_weights = df[["inv_id", "cmb_id", "total_kw_dc", "dc_weight", "dc_weight_adj"]]

    def __str__(self):
        row_list = []
        addrow = lambda str_: row_list.append(str_)
        addrow(f"Site Name: {self.site}")

        addrow("Metadata:")
        addrow(f"n_inverters = {len(self.inverter_names)}")
        for inv in self.inverter_names[:3]:
            addrow(f"   >> {inv}")
        addrow("   ...")

        addrow(f"n_cmb_atts = {len(self.cmb_att_names)} (not inverter-specific)")
        for att in self.cmb_att_names[:3]:
            addrow(f"   >> {att}")
        addrow("   ...")

        addrow(f"n_total_cmb_attributes = {len(self.attribute_paths)}")
        for ap in self.attribute_paths[:3]:
            addrow(f"   >> ..{ap.split('Solar Assets')[-1]}")
        addrow("   ...")

        addrow("self.rename_cols = ")
        for col_before, col_after in list(self.rename_cols.items())[:3]:
            addrow(f"   {col_before} :: {col_after}")
        addrow("   ...")

        addrow("self.id_mapping = ")
        for inv, dict_ in list(self.id_mapping.items())[:2]:
            addrow(f"   {inv}")
            for cmb_id, cmb_name in list(dict_.items())[:2]:
                addrow(f"      {cmb_id} :: {cmb_name}")
            addrow("      ...")
        addrow("   ...")
        return "\n".join(row_list)


CMB_ANALYSIS_DATA_KEYS = ["dtn", "combiners", "inverters", "met_stations", "curtailment"]
CMB_ANALYSIS_THRESHOLDS = {
    "offline_cmb_range": 1,  # current (amps) -- max. range of data in column
    "offline_inv": 10,  # min. total kwh over range to determine if inv is offline
    "underperform": 0.75,  # actual < 65% of possible
    "comms": 5,  # current (amps); used to determine offline vs. comms (if offline_cmb_range flag)
    "calibration_offset": 20,  # current (amps) -- e.g. combiner showing non-zero data overnight
    "tracker_stow_range": 50,  # current (amps) -- the higher, the less false positives -- min delta between afternoon and morning current to indicate a stowed tracker
    "curtailment_flag": 0,  # flag > 0 (from OE.CurtailmentFlag) - min. flag value to treat as curtailment
    "pct_curtailed": 0.625,  # if above this %, treat as curtailed day (applied to range where pvl > 0)
}
TRACKER_HOURS = (10, 12, 14)
COMBINER_FLAGS = [
    "offline",
    "underperform",
    "comms",
    "calibration",
    "curtailment",
    "tracker_east",
    "tracker_west",
    "tracker_center",
]


class DCHealthChecks:
    def __init__(self, site: str, data: dict[str, pd.DataFrame] = {}):
        self.meta = CombinerMetadata(site=site)
        self.site = site
        self.date_range = None  # [start_date, end_date]
        self.datetime_index = None
        self._process_input_data(data)

        self.flagged_combiners = {}  # updated in run_combiner_analysis
        self.thresholds = CMB_ANALYSIS_THRESHOLDS  # can be updated in run_combiner_analysis

    def _process_input_data(self, data):
        self.data = {}
        compare_indexes = []
        for key in (*CMB_ANALYSIS_DATA_KEYS, "pvlib"):
            df = data.get(key, pd.DataFrame())
            if not type(df) is pd.DataFrame:
                raise TypeError(f"Provided {key} data is not a dataframe.")
            if not df.empty:
                compare_indexes.append(df.index.copy())
            if key == "combiners":
                df = df.rename(columns=self.meta.rename_cols)
            self.data[key] = df.copy()

        if len(compare_indexes) > 1:
            idx0 = compare_indexes[0]
            for idx in compare_indexes[1:]:
                if not all(idx == idx0):
                    raise ValueError("One or more dataframes have different datetime index.")
            self.datetime_index = idx0
            self._assign_or_validate_date_range(idx0)

    def _get_data_status(self, grp):
        return not self.data.get(grp, pd.DataFrame()).empty

    def _assign_or_validate_date_range(self, datetime_index):
        # assigns range or validates datetime index
        start = datetime_index.min().floor("D")
        end = datetime_index.max()
        if end.hour > 0:
            end = end.ceil("D")
        start_date, end_date = list(map(lambda x: x.strftime("%Y-%m-%d"), [start, end]))

        # assign if not already exists
        if self.date_range is None:
            self.date_range = [start_date, end_date]

        # otherwise, validate
        if self.date_range[0] != start_date or self.date_range[1] != end_date:
            raise ValueError("Invalid datetime index detected; does not match existing data.")

    def _evaluate_input_dates(self, start_date, end_date) -> list[str]:
        if start_date is not None and end_date is not None:
            return [start_date, end_date]
        if start_date is None and end_date is None:
            if self.date_range is None:
                raise ValueError("Must provide start and end dates.")
            return self.date_range
        raise ValueError("Must provide both start and end date.")

    def _get_data(self, start_date, end_date, data_key, freq="1h", method="summaries", q=True):
        try:
            if data_key == "dtn":
                df = SolarDataset.from_dtn(self.site, start_date, end_date, q=q)
                return df

            kwargs = dict(
                site_name=self.site,
                start_date=start_date,
                end_date=end_date,
                freq=freq,
                method=method,
                keep_tzinfo=True,
                q=q,
            )
            if data_key == "curtailment":
                site_af_path = f"{self.meta.attribute_paths[0].split(self.site)[0]}" + self.site
                attpaths = [f"{site_af_path}|OE.CurtailmentStatus"]
                dataset = PIDataset.from_attribute_paths(attribute_paths=attpaths, **kwargs)
            else:
                group = data_key.replace("_", " ").title()
                if data_key == "combiners":
                    kwargs.update(dict(n_segment=3))
                dataset = SolarDataset.from_defined_query_attributes(asset_group=group, **kwargs)
            return dataset.data

        except Exception as e:
            print(f"Error while running query for {data_key} data")
            return None

    def query_analysis_data(
        self,
        start_date=None,
        end_date=None,
        data_keys=CMB_ANALYSIS_DATA_KEYS,
        freq="1h",
        overwrite=False,
        q=True,
    ):
        qprint = quiet_print_function(q=q)
        start_date, end_date = self._evaluate_input_dates(start_date, end_date)
        data_keys = [k for k in CMB_ANALYSIS_DATA_KEYS if k in data_keys]
        if not data_keys:
            raise ValueError("No valid data keys specified.")

        for key in data_keys:
            if not self.data[key].empty:
                qprint(f"Found existing data for {key}.")
                if overwrite is False:
                    qprint(f">>> skipping ({overwrite=})")
                    continue
                qprint(f">>> re-querying ({overwrite=})")
            else:
                qprint(f"Querying data for {key}...")

            if key == "met_stations" and self.data["dtn"].empty:
                raise ValueError("Error: need DTN data for running met station backfill.")

            df = self._get_data(start_date, end_date, data_key=key, freq=freq, q=q)

            if key in ("inverters", "met_stations"):
                qprint("Running auto QC...")
                original_cols = list(df.columns)
                df = run_auto_qc(df, site=self.site)
                df = df[[c for c in original_cols if c in df.columns]]
                for i, c in enumerate(original_cols):
                    if c not in df.columns:
                        df.insert(i, c, np.nan)
                if key == "met_stations":
                    qprint("Running backfill...")
                    df, changes = backfill_meteo_data(
                        df_met=df, df_ext=self.data["dtn"], site=self.site
                    )

            if key == "combiners":
                df = df.rename(columns=self.meta.rename_cols)
            if df is None:
                qprint(f"Error for {key=}.")
                continue
            self._assign_or_validate_date_range(df.index)
            self.data.update({key: df})

        qprint(f"End data query.")
        return

    def _get_poa_for_pvlib(self, source):
        if source not in ("dtn", "sensors"):
            raise ValueError("Invalid poa source specified.")
        if source == "sensors":
            df_met = self.data["met_stations"]
            if any(c.startswith("OE.POA") for c in df_met.columns):
                if "Processed_POA" in df_met.columns:
                    poa_data = df_met["Processed_POA"]
                else:
                    poa_data = df_met.filter(like="OE.POA").mean(axis=1)
                return pd.Series(poa_data, name="POA")
            print("NOTE: specified sensors, but no POA data found; using DTN.")
        elif self.data["dtn"].empty:
            raise ValueError("Invalid requirements for pvlib - missing DTN data.")
        return pd.Series(self.data["dtn"]["poa_global"], name="POA_DTN")

    def run_pvlib(self, poa_source="sensors", overwrite=False, q=True):
        """Runs combiner-level pvlib model. Updates self.data["pvlib"] with result."""
        qprint = quiet_print_function(q=q)
        if not self.data["pvlib"].empty:
            qprint(f"Found existing pvlib data.")
            if overwrite is False:
                qprint(f">>> skipping ({overwrite=})")
                return
            qprint(f">>> re-running model ({overwrite=})")
        else:
            qprint(f"Running pvlib model...")

        try:
            poa_data = self._get_poa_for_pvlib(source=poa_source)
            df_pvl = run_pvlib_combiner_level(
                site=self.site, poa_data=poa_data, return_dict=False, q=q
            )
        except Exception as e:
            raise ValueError(f"Error while running pvlib model: {e}")

        self._assign_or_validate_date_range(df_pvl.index)
        self.data.update({"pvlib": df_pvl})

    def _validate_requirements(self):
        for data_key in ["combiners", "inverters", "pvlib"]:
            if self.data[data_key].empty:
                raise ValueError(f"Missing the following required data: {missing_}")
        return

    def run_combiner_analysis(self, printouts=False, custom_thresholds={}) -> None:
        self._validate_requirements()
        qprint = lambda msg: None if printouts is False else print(msg)

        # get inverter data
        df_inv = self.data["inverters"].copy()
        tz = str(df_inv.index.tzinfo)

        # get combiner data and rename columns, remove negatives
        # df_cmb = self.data["combiners"].rename(columns=self.meta.rename_cols).clip(lower=0)
        df_cmb = self.data["combiners"].clip(lower=0).copy()

        # get pvlib data as dictionary
        cmb_pvlib_data = pvlib_dataframe_to_dict(self.data["pvlib"])

        # get dc_weights
        dc_weights = self.meta.dc_weights

        # join with combiner data and create possible kw cols for each cmb
        final_cmb_data = {}
        for inv_id, dfx in cmb_pvlib_data.items():
            dfw = dc_weights[dc_weights["inv_id"].eq(inv_id)].copy()
            for cmb_id, dc_wt in dfw[["cmb_id", "dc_weight_adj"]].values:
                cmb_col = f"possible_kw_{inv_id}_{cmb_id}"
                dfx[cmb_col] = dfx[f"possible_kw_{inv_id}"].mul(dc_wt)

            # combine & join with cmb query data
            final_cmb_data[inv_id] = dfx.join(df_cmb.filter(like=inv_id))

        # get analysis thresholds
        thresholds = {
            key: custom_thresholds.get(key, default) for key, default in self.thresholds.items()
        }

        # BEGIN ANALYSIS BY DAY
        analysis_dates = pd.date_range(*self.date_range, inclusive="left")

        # init output
        flagged_combiners_by_date = {d.strftime("%Y-%m-%d"): {} for d in analysis_dates}

        for inv_id, dfx_ in final_cmb_data.items():

            # add inv gen col to dfx
            inv_col = f"OE.ActivePower_{inv_id}"
            dfx_ = dfx_.join(df_inv[[inv_col]])

            # add curtailment data to dfx
            curtailment_flag_data = self.data["curtailment"]
            if curtailment_flag_data.empty:
                dfx_["curtailment_flag"] = 0
            else:
                dfx_["curtailment_flag"] = curtailment_flag_data["OE.CurtailmentStatus"].copy()

            # get cmb names from pvlib cols
            def get_cmb_name(imp_col):
                return imp_col.replace("i_mp_", "").replace(f"_{inv_id}", "")

            cmb_names = list(map(get_cmb_name, dfx_.filter(like="i_mp").columns))
            n_cmb = len(cmb_names)

            # assess dc performance by day
            for date in analysis_dates:
                date_str = date.strftime("%Y-%m-%d")
                dfx = dfx_[dfx_.index.date == date.date()].copy()
                dfx_online = dfx[dfx[inv_col].gt(0)].copy()
                if dfx_online.empty:
                    continue  # if inv was offline for the entire day

                # check inverter gen
                pvl_col = f"possible_kw_{inv_id}"
                inv_total_kwh = dfx_online[inv_col].sum()
                pvl_total_kwh = dfx_online[pvl_col].sum()

                # STOP CONDITION - inv not low enough relative to possible
                if inv_total_kwh > pvl_total_kwh * 0.95:
                    continue

                # create average curt flag for CURTAILMENT CHECK (used in cmb loop)
                avg_curt_flag = dfx.loc[dfx[pvl_col].gt(0), "curtailment_flag"].mean()
                if pd.isna(avg_curt_flag):
                    avg_curt_flag = 0

                # loop through combiners and check performance
                # cmb_dict = dict.fromkeys(COMBINER_FLAGS, {})
                cmb_dict = {flag_: {} for flag_ in COMBINER_FLAGS}
                for cmb in cmb_names:

                    # get possible kw for the combiner (for loss calc)
                    cmb_poss_kw_col = f"possible_kw_{inv_id}_{cmb}"
                    if cmb_poss_kw_col not in dfx.columns:
                        # when rows were removed from the design db due to a mismatch with pi atts
                        qprint(f"Note: skipping {inv_id=}, {cmb} b/c removed during init.")
                        continue

                    act_col = f"{cmb}_{inv_id}"  # cmb current
                    poss_col = f"i_mp_{act_col}"
                    if act_col not in dfx.columns:
                        # no data returned from pi query (not in df_cmb)
                        qprint(f"Note: skipping {inv_id=}, {cmb} b/c not found in pi query file.")
                        continue

                    # get total possible mwh for cmb, then adjust for curtailment using avg flag
                    cmb_poss_no_curtailment = dfx_online[cmb_poss_kw_col].sum() / 1e3
                    cmb_total_possible_mwh = cmb_poss_no_curtailment * (1 - avg_curt_flag)

                    # CURTAILMENT CHECK
                    if avg_curt_flag > thresholds["pct_curtailed"]:
                        cmb_dict["curtailment"].update({cmb: cmb_total_possible_mwh})
                        continue

                    # sum of cmb current (amps)
                    act_total, poss_total = dfx_online[act_col].sum(), dfx_online[poss_col].sum()

                    # CHECK 1 - OFFLINE/COMMS CHECKS (for flatlined data)
                    col_range = dfx_online[act_col].max() - dfx_online[act_col].min()
                    is_flatlined = col_range < thresholds["offline_cmb_range"]
                    if is_flatlined:
                        if act_total > thresholds["comms"]:
                            # COMMS OUTAGE
                            cmb_dict["comms"].update({cmb: 0})
                        else:
                            # OFFLINE
                            cmb_total_loss_mwh = cmb_total_possible_mwh
                            cmb_dict["offline"].update({cmb: cmb_total_loss_mwh})
                        continue

                    # temp helper function for error handling empty arrays
                    def getval(series):
                        if len(series.values) == 1:
                            return series.values[0]
                        return None

                    # CHECK 2 - CALIBRATION CHECKS
                    midnight_val = getval(dfx.loc[dfx.index.hour == 1, act_col])
                    if midnight_val:
                        if midnight_val > thresholds["calibration_offset"]:
                            # CALIBRATION OFFSET DETECTED
                            cmb_dict["calibration"].update({cmb: 0})
                            continue

                    # CHECK 3 - UNDERPERFORM CHECK
                    delta_pct = act_total / poss_total
                    cmb_total_loss_mwh = cmb_total_possible_mwh * (1 - delta_pct)

                    if delta_pct < thresholds["underperform"]:

                        # CHECK 3a - TRACKERS
                        if all(hr in dfx_online.index.hour for hr in TRACKER_HOURS):
                            am_value_act = getval(
                                dfx_online.loc[dfx_online.index.hour == TRACKER_HOURS[0], act_col]
                            )
                            noon_value_act = getval(
                                dfx_online.loc[dfx_online.index.hour == TRACKER_HOURS[1], act_col]
                            )
                            pm_value_act = getval(
                                dfx_online.loc[dfx_online.index.hour == TRACKER_HOURS[2], act_col]
                            )
                            if any(x is None for x in [am_value_act, noon_value_act, pm_value_act]):
                                continue

                            act_delta_pm_am = (
                                pm_value_act - am_value_act
                            )  # higher delta = bad tracker
                            if abs(act_delta_pm_am) > thresholds["tracker_stow_range"]:
                                if act_delta_pm_am > 0:
                                    flag_ = "tracker_west"
                                else:
                                    flag_ = "tracker_east"
                                cmb_dict[flag_].update({cmb: cmb_total_loss_mwh})
                                continue

                            # check for center stow - TODO
                            am_value_poss = getval(
                                dfx_online.loc[dfx_online.index.hour == TRACKER_HOURS[0], poss_col]
                            )
                            noon_value_poss = getval(
                                dfx_online.loc[dfx_online.index.hour == TRACKER_HOURS[1], poss_col]
                            )
                            pm_value_poss = getval(
                                dfx_online.loc[dfx_online.index.hour == TRACKER_HOURS[2], poss_col]
                            )
                            if any(
                                x is None for x in [am_value_poss, noon_value_poss, pm_value_poss]
                            ):
                                continue

                            am_delta = am_value_poss - am_value_act
                            noon_delta = noon_value_poss - noon_value_act
                            pm_delta = pm_value_poss - pm_value_act

                            am_cond = am_delta > thresholds["tracker_stow_range"]
                            pm_cond = pm_delta > thresholds["tracker_stow_range"]
                            noon_cond = noon_delta < thresholds["tracker_stow_range"]
                            if all([am_cond, pm_cond, noon_cond]):
                                cmb_dict["tracker_center"].update({cmb: cmb_total_loss_mwh})
                                continue

                        # if no tracker issues detected
                        cmb_dict["underperform"].update({cmb: cmb_total_loss_mwh})
                        continue

                if not any(cmb_dict.values()):
                    continue

                inv_flagged_cmb = {key: list_ for key, list_ in cmb_dict.items() if list_}
                flagged_combiners_by_date[date_str].update({inv_id: inv_flagged_cmb})

        self.thresholds = thresholds
        self._process_flagged_combiners(flagged_combiners_by_date)

    def _process_flagged_combiners(self, flagged_by_date):
        self.flagged_combiners_by_date = flagged_by_date
        if not flagged_by_date:
            self.flagged_combiners = {}
            return
        flagged_cmb_by_inv = {inv: {} for inv in self.meta.inverter_names}
        for date_str, dict_ in flagged_by_date.items():
            for inv, dict2_ in dict_.items():
                for flag, dict3_ in dict2_.items():
                    for cmb, loss in dict3_.items():
                        if flag not in flagged_cmb_by_inv[inv].keys():
                            flagged_cmb_by_inv[inv][flag] = {cmb: loss}
                        elif cmb not in flagged_cmb_by_inv[inv][flag].keys():
                            flagged_cmb_by_inv[inv][flag].update({cmb: loss})
                        else:
                            existing_loss = flagged_cmb_by_inv[inv][flag][cmb]
                            flagged_cmb_by_inv[inv][flag].update({cmb: existing_loss + loss})
        self.flagged_combiners = {
            inv: cmb_dict for inv, cmb_dict in flagged_cmb_by_inv.items() if cmb_dict
        }
        flagged_cmb_ids_by_inv = {inv: [] for inv in self.meta.inverter_names}
        for inv, dict_ in self.flagged_combiners.items():
            for flag, dict2_ in dict_.items():
                new_cmb = [c for c in dict2_.keys() if c not in flagged_cmb_ids_by_inv[inv]]
                flagged_cmb_ids_by_inv[inv].extend(new_cmb)
        self.flagged_cmb_ids = {
            inv: cmblist for inv, cmblist in flagged_cmb_ids_by_inv.items() if cmblist
        }
        return

    def get_dataframes_for_spreadsheet(self) -> dict[str, pd.DataFrame]:
        if not self.flagged_combiners:
            raise ValueError("No flagged combiners found.")

        # compiling for spreadsheet output (for first pass)
        df_cat = pd.DataFrame(
            index=self.meta.inverter_names, columns=self.meta.cmb_att_names, data=""
        )
        df_cat.index.name = self.site
        df_loss = df_cat.copy()  # init

        category_symbols = {
            "offline": "OFF",
            "underperform": "UND",
            "comms": "COMMS",
            "calibration": "CAL",
            "curtailment": "CURT",
            "tracker": "TRK",
        }

        def matching_symbol(category):
            for key, symbol in category_symbols.items():
                if category.startswith(key):
                    return symbol
            return

        for inv, cmb_dict in self.flagged_combiners.items():
            inv_cmb_mapping = self.meta.id_mapping[inv]  # {cmb_id: cmb_att_name}
            nonexistent_cmb_atts = [
                a for a in self.meta.cmb_att_names if a not in inv_cmb_mapping.values()
            ]
            for cmb_att_name in nonexistent_cmb_atts:
                df_cat.at[inv, cmb_att_name] = "-/-"
                df_loss.at[inv, cmb_att_name] = "-/-"

            for category, dict_ in cmb_dict.items():  # dict_ fmt {cmb_id: lost_mwh}
                sym = matching_symbol(category)
                if sym is None:
                    raise ValueError(f"invalid category {category}")
                for cmb_id, lost_mwh in dict_.items():
                    cmb_att_name = inv_cmb_mapping[cmb_id]  # column for output dfs
                    df_cat.at[inv, cmb_att_name] = sym
                    df_loss.at[inv, cmb_att_name] = round(lost_mwh, 4)

        return dict(categories=df_cat, lost_mwh=df_loss)

    def extract_relevant_data(self, inv, clip=True) -> dict:
        if any(self.data[key].empty for key in ["combiners", "inverters", "pvlib"]):
            raise KeyError("Missing data for one or more required groups.")

        # inv generation actual/possible
        dfi = self.data["pvlib"][[f"possible_kw_{inv}"]].join(
            self.data["inverters"][[f"OE.ActivePower_{inv}"]]
        )
        dfi.columns = [f"{inv}_possible_kw", f"{inv}_actual_kw"]

        # combiner current actual/possible
        dfc = self.data["combiners"].filter(like=inv).copy()
        dfc = dfc.rename(columns={c: c.replace(f"_{inv}", "_actual") for c in dfc.columns})

        dfp = self.data["pvlib"].filter(regex=f"i_mp.*{inv}").copy()
        dfp = dfp.rename(
            columns={c: c.replace("i_mp_", "").replace(f"_{inv}", "_possible") for c in dfp.columns}
        )

        cmb_ids = [c.replace("_actual", "") for c in dfc.columns]
        weights = self.meta.dc_weights.set_index("inv_id").loc[inv]

        df = pd.concat([dfi, dfc, dfp], axis=1)
        if clip is True:
            df = df.clip(lower=0.0)
        for cmb in cmb_ids:
            wt = weights.set_index("cmb_id").at[cmb, "dc_weight"]
            df[f"{cmb}_adjusted"] = df[f"{cmb}_actual"].div(wt)

        return {
            "df": df,
            "cmb_ids": cmb_ids,
            "weights": weights,
        }

    def flagged_cmb_by_date(self, inv) -> dict:
        if not self.flagged_combiners_by_date:
            return {}
        return {
            key: dict_[inv]
            for key, dict_ in self.flagged_combiners_by_date.items()
            if inv in dict_.keys()
        }
