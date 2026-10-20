import pandas as pd
from pathlib import Path

from .config import CACHE_DIR
from .helpers import quiet_print_function
from .oemeta import PI_SITES_BY_FLEET
from .oepaths import latest_file, validated_savepath

CACHE_FILE_TYPES = [
    "inverters",
    "met_stations",
    "meter",
    "site",
    "combiners",
    "dtn",
    "pvlib",
    "pvlib_cmb",
    "ghi",  # temp
]


def _range_id(start_date: str, end_date: str) -> str:
    """input dates should be formatted '%Y-%m-%d' -> example output: '20260128-20260211'"""
    start, end = map(pd.Timestamp, [start_date, end_date])
    return f"{start:%Y%m%d}-{end:%Y%m%d}"


def dates_from_range_id(range_id: str) -> list[str]:
    fmt_date_str = lambda x: "-".join([x[:4], x[4:6], x[6:8]])
    id_parts = range_id.split("-")
    if len(id_parts) == 1:
        id_parts.append("")
    return list(map(fmt_date_str, id_parts[:2]))


def _site_str(site: str) -> str:
    """examples: 'Adams East' -> 'adams-east'; 'CW-Goose Lake' -> 'cw-goose-lake'"""
    return site.lower().replace(" ", "-")


def infer_site_name(site_str) -> str:
    all_valid_pi_sites = [s for list_ in PI_SITES_BY_FLEET.values() for s in list_]
    for site in all_valid_pi_sites:
        if _site_str(site) == site_str:
            return site
    raise ValueError(f"No matching site found for {site_str=}.")


def generate_filestem(site: str, start_date: str, end_date: str, file_type: str) -> str:
    """example for site='Adams East', file_type='met_stations'
    -> filestem: 'adams-east_met_stations_20260128-20260211'
    """
    if file_type not in CACHE_FILE_TYPES:
        raise ValueError(f"{file_type=} is invalid.\nMust be one of: {CACHE_FILE_TYPES}")
    return "_".join([_site_str(site), file_type, _range_id(start_date, end_date)])


def extract_metadata_from_filestem(filestem: str) -> dict:
    parts = filestem.split("_")
    if len(parts) < 3:
        return {}
    site_str, *ftype_parts, range_id = parts
    site = infer_site_name(site_str=site_str)
    file_type = "_".join(ftype_parts)
    range_id = range_id[:17]  # in case of duplicate files, e.g. (1)
    start_date, end_date = dates_from_range_id(range_id=range_id)
    return dict(
        site=site,
        file_type=file_type,
        range_id=range_id,
        start_date=start_date,
        end_date=end_date,
    )


class DataCache:

    def __init__(self, folder: Path):
        self._validate_cache_dir(folder=folder)

    def _validate_cache_dir(self, folder):
        # validation of input cache folder
        if not isinstance(folder, Path):
            raise TypeError("Input 'folder' must be a Path object.")
        if not folder.exists():
            raise ValueError("Could not find specified folder.")
        self.folder = folder

        metadata_list = []
        for fpath in folder.glob("*.parquet"):
            meta = extract_metadata_from_filestem(fpath.stem)
            if meta == {}:
                continue  # invalid/unsupported filename
            metadata_list.append({"filepath": fpath, **meta})

        self.meta = metadata_list
        return

    def _filtered_metadata(self, **matching_items):
        if len(matching_items) == 0:
            return []
        is_match = lambda item: all(item[key] == val for key, val in matching_items.items())
        return list(filter(is_match, self.meta))

    def _extract_metadata(self, key: str, meta_list: list = None):
        mdata_list = self.meta if meta_list is None else meta_list
        return list(map(lambda item: item.get(key, None), mdata_list))

    @property
    def sites_with_data(self):
        return sorted(set(self._extract_metadata(key="site")))

    @property
    def filepaths_by_site(self):
        return {
            s: self._extract_metadata(key="filepath", meta_list=self._filtered_metadata(site=s))
            for s in self.sites_with_data
        }

    @property
    def filepaths_by_range_id_by_site(self):
        return {
            site: {
                range_id: sorted(
                    set(
                        self._extract_metadata(
                            key="filepath",
                            meta_list=self._filtered_metadata(site=site, range_id=range_id),
                        )
                    )
                )
                for range_id in rng_id_list
            }
            for site, rng_id_list in self.range_ids_by_site.items()
        }

    @property
    def file_types_by_site(self):
        return {
            site: sorted(
                set(
                    self._extract_metadata(
                        key="file_type",
                        meta_list=self._filtered_metadata(site=site),
                    )
                )
            )
            for site in self.sites_with_data
        }

    @property
    def range_ids_by_site(self):
        return {
            site: sorted(
                set(
                    self._extract_metadata(
                        key="range_id",
                        meta_list=self._filtered_metadata(site=site),
                    )
                )
            )
            for site in self.sites_with_data
        }

    @property
    def range_ids_by_file_type_by_site(self):
        return {
            site: {
                file_type: sorted(
                    set(
                        self._extract_metadata(
                            key="range_id",
                            meta_list=self._filtered_metadata(site=site, file_type=file_type),
                        )
                    )
                )
                for file_type in ftype_list
            }
            for site, ftype_list in self.file_types_by_site.items()
        }

    @property
    def file_types_by_range_id_by_site(self):
        return {
            site: {
                range_id: sorted(
                    set(
                        self._extract_metadata(
                            key="file_type",
                            meta_list=self._filtered_metadata(site=site, range_id=range_id),
                        )
                    )
                )
                for range_id in rng_id_list
            }
            for site, rng_id_list in self.range_ids_by_site.items()
        }

    def _validate_inputs(self, site: str, start_date: str, end_date: str) -> dict[str, Path]:
        if site not in self.filepaths_by_site:
            # raise ValueError(f"No cached files found for {site = }.")
            return {}

        range_id = _range_id(start_date, end_date)
        if range_id not in self.range_ids_by_site[site]:
            # raise ValueError(f"No data exists for {range_id=}.")
            return {}

        return self._cached_filepaths(site=site, range_id=range_id)

    def _cached_filepaths(self, site: str, range_id: str) -> dict[str, Path]:
        cached_fpaths = self.filepaths_by_range_id_by_site[site][range_id]
        fpath_dict = {}  # init
        for file_type in CACHE_FILE_TYPES:
            fpath = latest_file(
                list(
                    filter(
                        lambda fp: file_type
                        == extract_metadata_from_filestem(fp.stem)["file_type"],
                        cached_fpaths,
                    )
                )
            )
            if fpath is None:
                continue
            fpath_dict.update({file_type: fpath})
        return fpath_dict

    def load(
        self, site: str, start_date: str, end_date: str, q: bool = True
    ) -> dict[str, pd.DataFrame]:
        qprint = quiet_print_function(q=q)

        cached_fpaths = self._validate_inputs(site=site, start_date=start_date, end_date=end_date)
        data = {}  # init
        for file_type, fpath in cached_fpaths.items():
            data[file_type] = pd.read_parquet(fpath)
            qprint(f"Loaded {file_type} data: ..\\" + "\\".join(fpath.parts[-2:]))
        return data

    def generate_cache_filepath(
        self, site: str, start_date: str, end_date: str, file_type: str
    ) -> Path:
        filestem = generate_filestem(site, start_date, end_date, file_type)
        savepath = CACHE_DIR / f"{filestem}.parquet"
        return validated_savepath(savepath)
