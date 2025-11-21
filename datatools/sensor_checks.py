import itertools
import numpy as np
import pandas as pd

# from ..datatools.backfill import
from ..utils.assets import SolarSite
from ..utils.reporting import (
    FlashReportGenerator,
    get_start_and_end_dates,
    validate_reporting_period,
)
from ..utils.solar import SolarDataset


SENSOR_TYPES = ["POA", "Ambient_Temp", "Module_Temp", "Wind_Speed"]

OE_SENSOR_COLUMNS = [f"OE.{s_type}" for s_type in SENSOR_TYPES]

KNOWN_BAD_SENSOR_COLUMNS = {
    "GA4": "OE.POA_B3P03",
}


def column_from_attribute_path(attribute_path: str) -> str:
    asset_and_att_name = attribute_path.split("\\")[-1].split("|")
    return "_".join(reversed(asset_and_att_name))  # attribute_asset


def process_specified_date_range(start_date, end_date):
    start, end = map(pd.Timestamp, [start_date, end_date])
    if start > end:
        start, end = end, start
    if end > pd.Timestamp.now():
        raise ValueError("Some or all of specified date range hasn't happened yet.")

    specified_range = pd.date_range(start, end)
    reporting_periods = list(set((x.year, x.month) for x in specified_range))
    if len(reporting_periods) > 2:
        raise ValueError("Date ranges across more than 2 unique year/month currently unsupported")
    existing_report_periods = []
    for year, month in reporting_periods:
        try:
            validate_reporting_period(year, month)
            existing_report_periods.append((year, month))
        except ValueError:
            continue
    if not existing_report_periods:
        return dict(file_range=None, query_range=[start_date, end_date])

    existing_report_periods = [(2025, 10)]

    min_period_start = pd.Timestamp(*min(existing_report_periods), 1)
    max_period_end = pd.Timestamp(*max(existing_report_periods), 1) + pd.DateOffset(months=1)

    if min_period_start < start:
        min_period_start = start
    if max_period_end > end:
        max_period_end = end

    file_range = list(map(lambda x: x.strftime("%Y-%m-%d"), [min_period_start, max_period_end]))
    if max_period_end == end:
        query_range = None
    else:
        query_range = [file_range[1], end_date]
    return dict(file_range=file_range, query_range=query_range)


class SensorJudge(SolarSite):
    """Class for assessment of solar site sensor quality."""

    def __init__(self, site: str):
        super().__init__(site)  # raises error if site not in PI solar fleet
        if "Met Stations" not in self.asset_groups:
            raise ValueError("No Met Stations found.")

        self.met_stations = self.asset_names_by_group["Met Stations"]
        self.all_possible_sensor_columns = list(
            map("_".join, itertools.product(OE_SENSOR_COLUMNS, self.met_stations))
        )

    @property
    def sensor_attribute_paths(self) -> list[str]:
        """Returns a list of all associated Met Station attributes that exist in PI."""
        return self.get_reporting_query_attributes("Met Stations", validated=True)

    @property
    def expected_sensor_columns(self) -> list[str]:
        """Returns list of column names corresponding to sensor attribute paths."""
        return list(map(column_from_attribute_path, self.sensor_attribute_paths))

    def _validated_sensor_types(sensor_types: list[str]) -> list[str]:
        if not sensor_types:
            return SENSOR_TYPES
        validated_sensor_types = [s for s in sensor_types if s in SENSOR_TYPES]
        if len(validated_sensor_types) == 0:
            raise ValueError("No valid sensor types specified.")
        return validated_sensor_types

    def _load_sensor_data(self, start_date, end_date, sensor_types=[]):
        """Note: currently only supports up to 2 months of data. (only 2 unique year/month)"""
        sensor_types = self._validated_sensor_types(sensor_types)

        # check for existing query files
        reporting_periods = list(
            set((x.year, x.month) for x in pd.date_range(start_date, end_date))
        )


def check_sensor_data(site, start_date, end_date, sensor_types="all"):
    """Loads (or queries) Met Station data for specified period and compares with DTN equivalent"""
