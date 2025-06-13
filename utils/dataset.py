from abc import ABC, abstractmethod
import pandas as pd


class Dataset(ABC):
    """Base class for several types of datasets (e.g. InverterDataset, WeatherDataset, etc.)

    Attributes
    ----------
    data : pandas.DataFrame
        Time series data for dataset with index of type pandas.DatetimeIndex

    """

    def __init__(self):
        self.data = self.empty_dataframe()

    @property
    def start_datetime(self):
        """Earliest timestamp in dataset"""
        if self.data.empty:
            return pd.NaT
        return self.data.index.min()

    @property
    def end_datetime(self):
        """Latest timestamp in dataset"""
        if self.data.empty:
            return pd.NaT
        return self.data.index.max()

    @property
    @abstractmethod
    def columns(self):
        """List of columns to produce empty dataframe"""
        pass  # implemented by subclass

    def empty_dataframe(self):
        """Produces an empty dataframe for the given data format"""
        return pd.DataFrame(columns=self.columns)
