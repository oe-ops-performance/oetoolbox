import pandas as pd

from ..datatools.backfill import load_supporting_data
from ..datatools.pvlib import run_pvlib_model
from ..reporting.insurance_bi import get_inverter_outage_days
from ..utils.assets import SolarSite
from ..utils.pi import PIDataset
from ..utils.solar import SolarDataset


KPI_QUERY_ATTRIBUTES = [
    "OE.AvgPOA",
    "OE.CurtailmentStatus",
    "OE.MeterMW",
    "OE.ModulesOffline",
    "OE.SiteSetpoint",
]

# goal: create function "get_intra_month_estimates"
# class SolarKPIs:
#     def __init__(self, site_name: str, start_date: str, end_date: str)


def get_data_for_kpis(
    site: str, start_date: str, end_date: str, q: bool = True
) -> dict[str, pd.DataFrame]:
    """Function to retrieve data for calculating intra-month kpis."""
    output_dict = {"site_name": site}

    start, end = map(pd.Timestamp, [start_date, end_date])
    n_days = (end - start).days

    # pi data for site-level attributes
    kwargs = dict(
        site_name=site,
        start_date=start_date,
        end_date=end_date,
        freq="1h",
        keep_tzinfo=True,
        q=q,
    )
    dataset1 = PIDataset.from_attribute_paths(**kwargs, attribute_paths=KPI_QUERY_ATTRIBUTES)
    output_dict["site"] = dataset1.data.copy()

    # pi dataset for inverter-level attributes
    dataset2 = SolarDataset.from_defined_query_attributes(**kwargs, asset_group="Inverters")
    df_inv = dataset2.data.div(1e3).copy()  # converted to MW
    df_inv["Site_Actual"] = df_inv[df_inv.filter(like="ActivePower").columns].sum(axis=1)
    output_dict["inverters"] = df_inv.copy()

    # insurance bi / claimed outages
    output_dict["insurance_bi"] = get_inverter_outage_days(site, start_date, end_date)

    # external data from DTN, etc.
    dfx = load_supporting_data(site, start_date, end_date)
    output_dict["dtn"] = dfx.copy()

    # modeled possible power data from pvlib
    poa_data = pd.Series(dfx["poa_global"], name="POA_DTN")
    df_pvl = run_pvlib_model(site, datetime_index=dfx.index, poa_data=poa_data, q=q)
    df_pvl = df_pvl.div(1e3)  # converted to MW
    df_pvl["Site_Possible"] = df_pvl[df_pvl.filter(like="Possible_Power").columns].sum(axis=1)
    output_dict["pvlib"] = df_pvl.copy()

    return output_dict
