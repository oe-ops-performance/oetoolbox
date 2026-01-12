import numpy as np
import pandas as pd

# from ..datatools.pvlib import run_pvlib_model
# from ..reporting.insurance_bi import get_inverter_outage_days
# from ..utils.assets import SolarSite
# from ..utils.pi import PIDataset
# from ..utils.solar import SolarDataset
from ..utils.helpers import quiet_print_function


REPORT_KPI_COLUMN_MAPPING = {
    "DTN Insolation [kWh/m^2]": "DTN POA Insolation (kWh/m2)",
    "Insolation [kWh/m^2]": "POA Insolation (kWh/m2)",
    "GHI Insolation [kWh/m^2]": "GHI Insolation (kWh/m2)",
    "Possible Generation": "Possible Generation (MWh)",
    "Actual Generation": "Inverter Generation (MWh)",
    "Meter Generation": "Meter Generation (MWh)",
    "AC Module Loss [MWh]": "AC Module Loss (MWh)",
    # "Tracker Loss [MWh]": "Tracker Loss (MWh)",
    "Soiling Loss [MWh]": "Soiling Loss (MWh)",
    "DC / System Health": "DC/System Health Loss (MWh)",
    "Downtime": "Downtime Loss (MWh)",
    "Curtailment": "Curtailment - Non_Compensable (MWh)",
    "Inverter Availability": "Inverter Uptime Availability (%)",
}

ORDERED_KPI_COLUMNS = [  # order for tracker & kpis_and_notes files
    "DTN POA Insolation (kWh/m2)",
    "POA Insolation (kWh/m2)",
    "GHI Insolation (kWh/m2)",
    "Possible Generation (MWh)",
    "Inverter Generation (MWh)",
    "Meter Generation (MWh)",
    "Meter Generation - ADJUSTED (MWh)",
    "AC Module Loss (MWh)",
    "Tracker Loss (MWh)",
    "Soiling Loss (MWh)",
    "DC/System Health Loss (MWh)",
    "Snow Derate Loss (MWh)",
    "Downtime Loss (MWh)",
    "Curtailment - Compensable (MWh)",
    "Curtailment - Non_Compensable (MWh)",
    "Curtailment - Total (MWh)",
    "Insurance BI Adjustment (MWh)",
    "Inverter Uptime Availability (%)",
]

KPI_COLUMNS_FOR_PLOT = [
    "Budgeted POA (kWh/m2)",
    "POA Insolation (kWh/m2)",
    "Budgeted Production (MWh)",
    "Possible Generation (MWh)",
    "Inverter Generation (MWh)",
    "Meter Generation (MWh)",
    "Budgeted Curtailment (MWh)",
    "Curtailment - Total (MWh)",
    "DC/System Health Loss (MWh)",
    "Snow Derate Loss (MWh)",
    "Downtime Loss (MWh)",
    "Insurance BI Adjustment (MWh)",
    "Inverter Uptime Availability (%)",
]


def blank_kpi_dataframe():
    return pd.DataFrame({col: np.nan for col in ORDERED_KPI_COLUMNS}, index=[0])


KPI_QUERY_ATTRIBUTES = [
    "OE.AvgPOA",
    "OE.CurtailmentStatus",
    "OE.MeterMW",
    "OE.ModulesOffline",
    "OE.SiteSetpoint",
]

# the following functions are designed to work with "kpi_data" output from FlashReportGenerator._get_performance_breakdown


# validate totals
def get_delta(x):
    possible = x["generation"]["possible"]
    total_losses = sum(x["losses"].values())
    total_adjustments = sum(x["adjustments"].values())
    net = possible - total_losses - total_adjustments
    actual = x["generation"]["meter"]
    return net - actual


# function to validate kpi totals
def validate_kpi_totals(kpi_data, return_delta=False):
    delta = get_delta(kpi_data)
    valid = round(delta, 5) == 0.0
    if return_delta is False:
        return valid
    return valid, delta


# function to check kpi totals and attempt to reconcile losses (if overestimated)
def reconcile_losses(kpi_data, q=True) -> None:
    """Uses kpi_data output from FlashReportGenerator._get_performance_breakdown."""
    qprint = quiet_print_function(q=q)
    _, delta = validate_kpi_totals(kpi_data, return_delta=True)
    if delta >= 0:
        return
    # if delta is negative, we are over-estimating our losses

    # check for insurance bi adjustment; if any, try removing from downtime
    insurance_adj = kpi_data["adjustments"]["insurance_bi"]
    downtime_loss = kpi_data["losses"]["downtime"]
    if insurance_adj > 0 and downtime_loss > 0:
        new_downtime = max(downtime_loss - insurance_adj, 0)
        kpi_data["losses"]["downtime"] = new_downtime
        if not q:
            qprint(
                ">> removed insurance_bi from downtime loss; "
                f"reduced from {downtime_loss:.2f} to {new_downtime:.2f}"
            )
        _, delta = validate_kpi_totals(kpi_data, return_delta=True)
        if delta >= 0:
            return

    # order of preference for removing losses from category
    ordered_losses = ["dc_health", "downtime", "soiling", "snow_derate", "ac_module", "curtailment"]
    for loss_type in ordered_losses:
        loss = kpi_data["losses"][loss_type]
        if loss > 0:
            loss_reduction = min(loss, delta * -1)  # note: delta is negative
            new_loss = loss - loss_reduction
            kpi_data["losses"][loss_type] = new_loss
            if not q:
                qprint(
                    f">> reduced {loss_type} loss by {loss_reduction:.2f} "
                    f"(new value: {new_loss:.2f})"
                )
            delta += loss_reduction
            if delta >= 0:
                break

    return


def format_kpi_data_for_waterfall(kpi_data: dict):
    generation = kpi_data["generation"]
    losses = {key: val * -1 for key, val in kpi_data["losses"].items()}
    adjustments = {key: val * -1 for key, val in kpi_data["adjustments"].items()}
    waterfall_data = {
        "possible": generation["possible"],
        "snow_derate_loss": losses["snow_derate"],
        "soiling_loss": losses["soiling"],
        "dc_health_loss": losses["dc_health"],
        "module_loss": losses["ac_module"],
        "downtime_loss": losses["downtime"],
        "insurance_bi": adjustments["insurance_bi"],
        "curtailment": losses["curtailment"],
        "inverter": generation["inverter"],
        "ac_line_losses": losses["ac_line_losses"],
        "meter": generation["meter"],
    }
    return waterfall_data


# goal: create function "get_intra_month_estimates"
# class SolarKPIs:
#     def __init__(self, site_name: str, start_date: str, end_date: str)


# def get_data_for_kpis(
#     site: str, start_date: str, end_date: str, q: bool = True
# ) -> dict[str, pd.DataFrame]:
#     """Function to retrieve data for calculating intra-month kpis."""
#     output_dict = {"site_name": site}

#     start, end = map(pd.Timestamp, [start_date, end_date])
#     n_days = (end - start).days

#     # pi data for site-level attributes
#     kwargs = dict(
#         site_name=site,
#         start_date=start_date,
#         end_date=end_date,
#         freq="1h",
#         keep_tzinfo=True,
#         q=q,
#     )
#     dataset1 = PIDataset.from_attribute_paths(**kwargs, attribute_paths=KPI_QUERY_ATTRIBUTES)
#     output_dict["site"] = dataset1.data.copy()

#     # pi dataset for inverter-level attributes
#     dataset2 = SolarDataset.from_defined_query_attributes(**kwargs, asset_group="Inverters")
#     df_inv = dataset2.data.div(1e3).copy()  # converted to MW
#     df_inv["Site_Actual"] = df_inv[df_inv.filter(like="ActivePower").columns].sum(axis=1)
#     output_dict["inverters"] = df_inv.copy()

#     # insurance bi / claimed outages
#     output_dict["insurance_bi"] = get_inverter_outage_days(site, start_date, end_date)

#     # external data from DTN, etc.
#     dfx = load_supporting_data(site, start_date, end_date)  #TODO -- need function that gets DTN
#     output_dict["dtn"] = dfx.copy()

#     # modeled possible power data from pvlib
#     poa_data = pd.Series(dfx["poa_global"], name="POA_DTN")
#     df_pvl = run_pvlib_model(site, datetime_index=dfx.index, poa_data=poa_data, q=q)
#     df_pvl = df_pvl.div(1e3)  # converted to MW
#     df_pvl["Site_Possible"] = df_pvl[df_pvl.filter(like="Possible_Power").columns].sum(axis=1)
#     output_dict["pvlib"] = df_pvl.copy()

#     return output_dict
