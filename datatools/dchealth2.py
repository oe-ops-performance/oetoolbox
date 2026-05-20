"""
DCHealthChecks2 - extends DCHealthChecks to accept a SolarAnalysisDataset as the
data source, eliminating the need to run separate PI queries for the DC health and
comparison analysis workflows.

Usage with SolarAnalysisDataset
--------------------------------
>>> ds = SolarAnalysisDataset.from_data_dict("FL1", data, start_date, end_date)
>>> ds.query_data(groups=["combiners", "inverters", "met_stations", "dtn", "curtailment"])
>>> ds.run_pvlib_cmb()
>>> dch = DCHealthChecks(data=ds)        # site name inferred from ds.site
>>> dch.run_pvlib(overwrite=False)       # skipped if pvlib_cmb already populated
>>> dch.run_combiner_analysis()

Backward-compatible usage (original dict-based API still works)
----------------------------------------------------------------
>>> dch = DCHealthChecks("FL1", data={"inverters": df_inv, ...})
"""

import pandas as pd

from .dchealth import DCHealthChecks as _DCHealthChecks


class DCHealthChecks(_DCHealthChecks):
    """DCHealthChecks extended to accept SolarAnalysisDataset as data source.

    Inherits all analysis methods from DCHealthChecks. Only __init__ is overridden
    to support SolarAnalysisDataset in addition to the original dict-based API.

    When a SolarAnalysisDataset is provided:
    - site name is inferred from dataset.site.name (can be overridden via site arg)
    - data["pvlib_cmb"] is mapped to data["pvlib"] for the combiner-level analysis
    - all other keys are passed through unchanged

    Parameters
    ----------
    site : str, optional
        Site name. Required when data is a plain dict. Inferred from the dataset
        when data is a SolarAnalysisDataset.
    data : dict[str, pd.DataFrame] or SolarAnalysisDataset
    """

    def __init__(self, site: str = None, data={}):
        from ..utils.solar2 import SolarAnalysisDataset

        if isinstance(data, SolarAnalysisDataset):
            if site is None:
                site = data.site.name
            input_data = dict(data.data)
            # combiner-level pvlib is stored under 'pvlib_cmb' in SolarAnalysisDataset;
            # DCHealthChecks expects it under 'pvlib'
            if "pvlib_cmb" in input_data:
                input_data["pvlib"] = input_data.pop("pvlib_cmb")
            data = input_data

        if site is None:
            raise ValueError("site must be provided when data is not a SolarAnalysisDataset.")

        super().__init__(site=site, data=data)
