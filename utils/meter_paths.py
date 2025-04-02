import calendar
import pandas as pd
from pathlib import Path

from ..reporting.tools import ALL_SOLAR_SITES, ALL_WIND_SITES
from ..utils import oepaths
from ..utils.helpers import quiet_print_function


UTILITY_DATA_OUTPUT_FOLDER = Path(oepaths.UTILITY_METER_DIR, "Output")

UTILITY_DATA_SOURCE_DIRS = {
    "solar": Path(oepaths.UTILITY_METER_DIR, "Solar"),
    "wind": Path(oepaths.UTILITY_METER_DIR, "Wind"),
}

PI_DATA_SITES = ["CW-Marin", "Indy I", "Indy II", "Indy III"]

VARSITY_SITE_IDS = {
    "Adams East": "ADMEST_6_SOLAR",
    "Alamo": "VICTOR_1_SOLAR2",
    "Camelot": "CAMLOT_2_SOLAR1",
    "Catalina II": "CATLNA_2_SOLAR2",
    "CID": "CORCAN_1_SOLAR1",
    "Columbia II": "CAMLOT_2_SOLAR2",
    "CW-Corcoran": "CORCAN_1_SOLAR2",
    "CW-Goose Lake": "GOOSLK_1_SOLAR1",
    "Kansas": "LEPRFD_1_KANSAS",
    "Kent South": "KNTSTH_6_SOLAR",
    "Maricopa West": "MARCPW_6_SOLAR1",
    "Old River One": "OLDRV1_6_SOLAR",
    "West Antelope": "ACACIA_6_SOLAR",
}


def last_day_of_month(year: int, month: int) -> int:
    """returns the last day of a given month"""
    return calendar.monthrange(year, month)[1]


def get_next_year_month(year: int, month: int) -> tuple:
    """returns the next year and month (as integers)"""
    next_period = pd.Timestamp(year, month, 1) + pd.DateOffset(months=1)
    return next_period.year, next_period.month


def month_name_and_abbr(month: int):
    return calendar.month_abbr[month], calendar.month_name[month]


def get_year_month_strings(year, month):
    """returns dictionary with helpful year/month keywords for use in file patterns"""
    yyyy = str(year)
    yy, mm = yyyy[-2:], f"{month:02d}"
    return yyyy, yy, mm


def get_legacy_meter_filepaths(year, month):
    """returns dictionary with site names as keys and filepath lists as values"""
    yyyy, yy, mm = get_year_month_strings(year, month)
    month_abbr, month_name = month_name_and_abbr(month)
    last_day = last_day_of_month(year, month)
    next_year, next_month = get_next_year_month(year, month)

    ATLAS_PATH = Path(oepaths.commercial, "Atlas Portfolio")
    VARSITY_PATH = Path(oepaths.varsity_metergen, yyyy, f"{month}_{yy}")
    ISONE_PATH = Path(oepaths.commercial, "ISO-NE", "FTP Files", "ISONE")
    ERCOT_PATH = Path(oepaths.commercial, "ERCOT", "Hedge Settlements", "Novatus Swap Models")

    solar_meter_folders = {
        "AZ1": Path(ATLAS_PATH, "1.05 AZ 1", "1.05.3 Meter Data"),
        "Comanche": Path(oepaths.commercial, "Comanche Invoices", yyyy, f"{mm}{yyyy}"),
        "CW-Marin": oepaths.frpath(year, month, ext="solar", site="CW-Marin"),
        "GA3": Path(ATLAS_PATH, "1.01 GA Solar 3", "1.01.06 Meter Data"),
        "GA4": Path(ATLAS_PATH, "1.02 GA Solar 4", "1.02.06 Meter Data"),
        "Grand View East": Path(ATLAS_PATH, "1.03 Grand View", "1.03.02 Meter Data"),
        "Grand View West": Path(ATLAS_PATH, "1.03 Grand View", "1.03.02 Meter Data"),
        "Indy I": oepaths.frpath(year, month, ext="solar", site="Indy I"),
        "Indy II": oepaths.frpath(year, month, ext="solar", site="Indy II"),
        "Indy III": oepaths.frpath(year, month, ext="solar", site="Indy III"),
        "Maplewood 1": Path(
            ATLAS_PATH, "1.08 Maplewood 1", "1.08.02 Tenaska", "1.08.02.1 Meter Data"
        ),
        "Maplewood 2": Path(
            ATLAS_PATH, "1.09 Maplewood 2", "1.09.02 Tenaska", "1.09.02.1 Meter Data"
        ),
        "MS3": Path(ATLAS_PATH, "1.07 MS 3", "1.07.02 Meter Data"),
        "Sweetwater": Path(ATLAS_PATH, "1.06 Sweetwater", "1.06.02 Invoices"),  # +draft/invoice
        "Three Peaks": Path(ATLAS_PATH, "1.04 Three Peaks", "1.04.01 Energy Invoices"),  # + "
    }

    atlas_file_patterns = {
        "AZ1": f"*AZ*Solar*{mm}-{yyyy}*",
        "GA3": f"*Solar*Generation*{yyyy}*{mm}*",
        "GA4": f"*Generation*{mm}*{yyyy}*",
        "Grand View East": f"*Solar*{month_abbr.upper()}*{yy}*",
        "Grand View West": f"*Solar*{month_abbr.upper()}*{yy}*",
        "Maplewood 1": f"*RealTimeEnergyDetails*{yyyy}-{mm}-01*-{last_day}*.xlsx",
        "Maplewood 2": f"*RealTimeEnergyDetails*{yyyy}-{mm}-01*-{last_day}*.xlsx",
        "MS3": f"*MSSL*{yy}{mm}*",
        "Sweetwater": f"*Solar*Invoice*{month_name}*{yyyy}*",  # + another folder
        "Three Peaks": f"*Power*Invoice*{month_name}*{yyyy}*",  # + another folder
    }

    varsity_file_patterns_1 = {
        "Azalea": f"*{mm}*{yyyy}*Azalea*Generation*",
        "Imperial Valley": f"*{mm}*IVSC*{yyyy}*",
        "Mulberry": f"2839*{month_abbr.upper()}*{yy}*",
        "Pavant": f"*Invoice*Pavant*Solar*{next_year}{next_month:02d}*",
        "Richland": f"*Richland*Generation*",
        "Selmer": f"2838*{month_abbr.upper()}*{yy}*",
        "Somers": f"*C*_hourly.csv",
    }

    varsity_file_patterns_2 = {site: "*stlmtui*" for site in VARSITY_SITE_IDS}

    varsity_file_patterns = varsity_file_patterns_1 | varsity_file_patterns_2

    varsity_meter_folders = {site: VARSITY_PATH for site in varsity_file_patterns}
    solar_meter_folders.update(varsity_meter_folders)

    comanche_file_pattern = {"Comanche": f"{yyyy}-{mm}*Comanche*"}
    pi_file_patterns = {site: "PIQuery_Meter*.csv" for site in PI_DATA_SITES}

    solar_file_patterns = (
        atlas_file_patterns
        | varsity_file_patterns
        | varsity_file_patterns_2
        | comanche_file_pattern
        | pi_file_patterns
    )

    wind_meter_folders = {
        "Bingham": Path(ISONE_PATH, "Bingham", "CMP_MeterReads_Monthly"),
        "Hancock": Path(ISONE_PATH, "Hancock", "Emera Meter Reads"),
        "Oakfield": Path(ISONE_PATH, "Oakfield", "Emera Meter Reads"),
        "Palouse": Path(oepaths.commercial, "PALOUSE INVOICES", yyyy),
        "Route 66": Path(ERCOT_PATH, "Rt66", "Tenaska Files", yyyy, f"{month_name} {yyyy}"),
        "South Plains II": Path(ERCOT_PATH, "SP2", "Tenaska Files", yyyy, f"{month_name} {yyyy}"),
        "Sunflower": Path(oepaths.commercial, "Sunflower Invoices", yyyy, f"{mm}{yyyy}"),
    }

    wind_file_patterns = {
        "Bingham": f"*Bingham*{month_name}_{yyyy}.xlsx",
        "Hancock": f"*Hancock*{month_name}_{yyyy}.xlsx",
        "Oakfield": f"*Oakfield*{month_name}_{yyyy}.xlsx",
        "Palouse": f"*{month_abbr}*{yyyy}*Gen.xlsx",
        "Route 66": "*Shadow*Real*Time*Energy*Imbalance*Detail*",
        "South Plains II": "*Shadow*Real*Time*Energy*Imbalance*Detail*",
        "Sunflower": "*SUN*.PRN",
    }

    meter_folders = solar_meter_folders | wind_meter_folders
    file_patterns = solar_file_patterns | wind_file_patterns

    output_dict = {}
    for site, folder in meter_folders.items():

        filepattern = file_patterns[site]

        if site in ["Three Peaks", "Sweetwater"]:
            draft_fp = list(folder.glob("*Draft Invoices"))[0]
            final_fp = list(folder.glob("*Final Invoices"))[0]
            mfileexists = lambda fp: (len(list(fp.glob(filepattern))) > 0)
            folder = final_fp if mfileexists(final_fp) else draft_fp

        meter_fpaths = list(folder.glob(filepattern))

        if site == "Comanche":
            isvalid = lambda fp: "Curtailment" not in fp.name and fp.suffix in [".xlsx", ".csv"]
            meter_fpaths = [fp for fp in meter_fpaths if isvalid(fp)]

        elif site in VARSITY_SITE_IDS:
            meter_fpaths = [fp for fp in meter_fpaths if (fp.suffix in [".xlsx", ".xls"])]
            unique_stems = list(set(fp.stem.replace(".xls", "") for fp in meter_fpaths))
            if len(unique_stems) < len(meter_fpaths):
                xlsx_fp = lambda stem_: Path(folder, f"{stem_}.xlsx")
                preferred_fpath = lambda s: (
                    xlsx_fp(s) if xlsx_fp(s).exists() else Path(folder, f"{s}.xls")
                )
                meter_fpaths = [preferred_fpath(stem_) for stem_ in unique_stems]

        output_dict[site] = oepaths.sorted_filepaths(meter_fpaths)

    return output_dict
