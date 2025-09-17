import pandas as pd

from ..utils.helpers import quiet_print_function
from ..utils.pi import PIDataset


def query_tracker_data(
    site: str,
    pipoint_list: list[str],
    start_date: str,
    end_date: str,
    freq: str = "24h",
    summary_type: str = "range",
    data_format: str = "wide",
    q: bool = True,
):
    "Runs PI query for list of pipoints (typically TiltAngleDegree or similar)"
    qprint = quiet_print_function(q=q)

    qprint(f"Begin tracker query for {site = }. (n_pipoints = {len(pipoint_list)})")
    kwargs = dict(
        site_name=site,
        start_date=start_date,
        end_date=end_date,
        method="summaries",
        freq=freq,
        data_format=data_format,
        n_segment=5,
        q=q,
        summary_type=summary_type,
    )

    chunk = 400
    split_pipoints = [pipoint_list[i : i + chunk] for i in range(0, len(pipoint_list), chunk)]

    df_list = []
    for i, sub_list in enumerate(split_pipoints):
        qprint(f"\nSublist {i+1} of {len(split_pipoints)}")
        dataset = PIDataset.from_pipoints(**kwargs, pipoint_names=sub_list)
        df_list.append(dataset.data)

    axis_ = 1 if data_format == "wide" else 0
    return pd.concat(df_list, axis=axis_)
