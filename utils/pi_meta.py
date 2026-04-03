import clr, sys

sys.path.append(r"C:\Program Files (x86)\PIPC\AF\PublicAssemblies\4.0")
clr.AddReference("OSIsoft.AFSDK")
from OSIsoft.AF import AFObject, PISystems  # type: ignore


class PIDatabase:
    def __init__(self, database_name: str = "Onward Energy"):
        self.pisystem = PISystems().DefaultPISystem
        self.name = database_name
        self._get_database_object()
        self.hierarchy = {}

    def _get_database_object(self):
        existing_dbs = [x.Name for x in self.pisystem.Databases]
        if self.name not in existing_dbs:
            raise KeyError(f"Name '{self.name}' is invalid. Existing databases: {existing_dbs}")
        self.db = self.pisystem.Databases.get_Item(self.name)

    def generate_hierarchy(self, fleet=None) -> dict:
        """Returns a dictionary with all existing element names (fleets, sites, groups, assets, etc.)
        -> note: directly from PI (i.e. not from meta JSON files)
        """
        if not self.name.startswith("Onward Energy"):
            print(f"Database {self.name} is currently not supported for this function.")
            return {}
        if str(fleet) not in ("None", "gas", "wind", "solar"):
            print(f"Fleet {fleet} is currently not supported for this function.")
            return {}

        if fleet is None:
            start_element = self.db
        else:
            database_path = self.db.GetPath()
            if "gas" in fleet.lower():
                fleet_path = f"{database_path}\\Gas Fleet"
            else:
                fleet_id = "Solar" if "solar" in fleet.lower() else "Wind"
                fleet_path = f"{database_path}\\Renewable Fleet\\{fleet_id} Assets"
            start_element = AFObject.FindObject(fleet_path)
            if start_element is None:
                raise Exception("Unknown error finding element for specified fleet/site.")

        hierarchy = {}

        # level 0 (top-level)
        for element_0 in start_element.Elements:
            #  site-level elements
            name_0 = element_0.Name
            if element_0.Elements.Count == 0:
                hierarchy[name_0] = []
                continue
            if not self._another_level_exists(element_0):
                hierarchy[name_0] = [e.Name for e in element_0.Elements]
                continue

            hierarchy_0 = {}

            # level 1
            for element_1 in element_0.Elements:
                #  asset group-level elements
                name_1 = element_1.Name
                if element_1.Elements.Count == 0:
                    hierarchy_0[name_1] = []
                    continue
                if not self._another_level_exists(element_1):
                    hierarchy_0[name_1] = [e.Name for e in element_1.Elements]
                    continue

                hierarchy_1 = {}

                # level 2
                for element_2 in element_1.Elements:
                    # asset-level elements
                    name_2 = element_2.Name
                    if element_2.Elements.Count == 0:
                        hierarchy_1[name_2] = []
                        continue
                    if not self._another_level_exists(element_2):
                        hierarchy_1[name_2] = [e.Name for e in element_2.Elements]
                        continue

                    hierarchy_2 = {}

                    # level 3
                    for element_3 in element_2.Elements:
                        # subasset-level elements
                        name_3 = element_3.Name
                        if element_3.Elements.Count == 0:
                            hierarchy_2[name_3] = []
                            continue
                        if not self._another_level_exists(element_3):
                            hierarchy_2[name_3] = [e.Name for e in element_3.Elements]
                            continue

                        hierarchy_3 = {}

                        # level 4
                        for element_4 in element_3.Elements:
                            # sub-subasset-level elements
                            name_4 = element_4.Name
                            if element_4.Elements.Count == 0:
                                hierarchy_3[name_4] = []
                                continue
                            if not self._another_level_exists(element_4):
                                hierarchy_3[name_4] = [e.Name for e in element_4.Elements]
                                continue

                            hierarchy_4 = {}

                            # level 5
                            for element_5 in element_4.Elements:
                                name_5 = element_5.Name
                                if element_5.Elements.Count == 0:
                                    hierarchy_4[name_5] = []
                                    continue
                                if not self._another_level_exists(element_5):
                                    hierarchy_4[name_5] = [e.Name for e in element_5.Elements]
                                    continue

                                hierarchy_5 = {}

                                # level 6
                                for element_6 in element_5.Elements:
                                    name_6 = element_6.Name
                                    if element_6.Elements.Count == 0:
                                        hierarchy_5[name_6] = []
                                        continue
                                    if not self._another_level_exists(element_6):
                                        hierarchy_5[name_6] = [e.Name for e in element_6.Elements]
                                    else:
                                        print("WARNING -- additional levels exist")

                                hierarchy_4[name_5] = hierarchy_5

                            hierarchy_3[name_4] = hierarchy_4

                        hierarchy_2[name_3] = hierarchy_3

                    hierarchy_1[name_2] = hierarchy_2

                hierarchy_0[name_1] = hierarchy_1

            hierarchy[name_0] = hierarchy_0

        self.hierarchy = hierarchy

    def _another_level_exists(self, element):
        """Returns True if any of the sub_elements have additional elements."""
        if element.Elements.Count == 0:
            return False
        return any(e.Elements.Count > 0 for e in element.Elements)


class PISiteElement:
    """A class for accessing PI Element metadata via AFSDK."""

    def __init__(self, fleet: str, site_name: str, q: bool = True):

        pisystem = PISystems().DefaultPISystem
        database = pisystem.Databases.get_Item("Onward Energy")
        if not any(x in fleet.lower() for x in ["gas", "solar", "wind"]):
            raise ValueError("Invalid fleet specified.")

        database_path = database.GetPath()
        if "gas" in fleet.lower():
            fleet_id = "Gas"
            fleet_path = f"{database_path}\\Gas Fleet"
        else:
            fleet_id = "Solar" if "solar" in fleet.lower() else "Wind"
            fleet_path = f"{database_path}\\Renewable Fleet\\{fleet_id} Assets"

        site_path = f"{fleet_path}\\{site_name}"
        site_element = AFObject.FindObject(site_path)
        if site_element is None:
            raise Exception("Unknown error finding element for specified fleet/site.")

        self.element = site_element  # OSIsoft.AF.Asset.AFElement
        self.site = site_name
        self.path = site_path
        self.fleet = fleet_id
        self._generate_asset_hierarchy()

    def _generate_asset_hierarchy(self, q: bool = True) -> dict:
        """Returns a dictionary with all existing element names (groups, assets, etc.)
        -> note: directly from PI (i.e. not from meta JSON files)
        """
        hierarchy = {}
        for group_element in self.element.Elements:
            group_name = group_element.Name
            if group_element.Elements.Count == 0:
                hierarchy[group_name] = []
                continue

            if not self._another_level_exists(group_element):
                hierarchy[group_name] = [e.Name for e in group_element.Elements]
                continue

            group_hierarchy = {}
            for element in group_element.Elements:
                if not self._another_level_exists(element):
                    group_hierarchy[element.Name] = [e.Name for e in element.Elements]
                    continue

                element_hierarchy = {}
                for sub_1 in element.Elements:
                    if not self._another_level_exists(sub_1):
                        element_hierarchy[sub_1.Name] = [e.Name for e in sub_1.Elements]
                        continue

                    sub_1_hierarchy = {}
                    for sub_2 in sub_1.Elements:
                        if not self._another_level_exists(sub_2):
                            sub_1_hierarchy[sub_2.Name] = [e.Name for e in sub_2.Elements]
                        else:
                            print("WARNING -- additional levels exist")

                    element_hierarchy[sub_1.Name] = sub_1_hierarchy

                group_hierarchy[element.Name] = element_hierarchy

            hierarchy[group_name] = group_hierarchy

        self.hierarchy = hierarchy

    def _another_level_exists(self, element):
        """Returns True if any of the sub_elements have additional elements."""
        if element.Elements.Count == 0:
            return False
        return any(e.Elements.Count > 0 for e in element.Elements)

    @property
    def asset_groups(self) -> list[str]:
        """Returns list of available asset groups."""
        return list(self.hierarchy.keys())

    def _validate_asset_group(self, asset_group):
        if asset_group not in self.asset_groups:
            raise KeyError("Invalid asset group.")
        return

    def get_asset_names(self, asset_group) -> list[str]:
        """Returns list of asset names corresponding to specified asset group."""
        self._validate_asset_group(asset_group)
        assets = self.hierarchy[asset_group]
        if type(assets) is list:
            return assets
        return list(assets.keys())

    def _validate_asset(self, asset_group, asset_name):
        self._validate_asset_group(asset_group)
        if asset_name not in self.hierarchy[asset_group]:
            raise KeyError("Invalid asset name.")
        return

    def get_sub_asset_names(self, asset_group, asset_name) -> list[str]:
        self._validate_asset(asset_group, asset_name)
        sub_assets = self.hierarchy[asset_group][asset_name]
        if type(sub_assets) is list:
            return sub_assets
        return list(sub_assets.keys())

    def _validate_sub_asset(self, asset_group, asset_name, sub_asset_name):
        self._validate_asset(asset_group, asset_name)
        if sub_asset_name not in self.hierarchy[asset_group][asset_name]:
            raise KeyError("Invalid sub-asset name.")
        return

    def get_sub_2_asset_names(self, asset_group, asset_name, sub_asset_name) -> list[str]:
        self._validate_sub_asset(asset_group, asset_name, sub_asset_name)
        sub_2_assets = self.hierarchy[asset_group][asset_name][sub_asset_name]
        if type(sub_2_assets) is list:
            return sub_2_assets
        return list(sub_2_assets.keys())

    def _validate_sub_2_asset(self, asset_group, asset_name, sub_asset_name, sub_2_asset_name):
        self._validate_sub_asset(asset_group, asset_name, sub_asset_name)
        if sub_2_asset_name not in self.hierarchy[asset_group][asset_name][sub_asset_name]:
            raise KeyError("Invalid sub-sub-asset name.")
        return

    def _get_element_path(self, asset_group=None, asset=None, sub_asset=None, sub_2_asset=None):
        path_parts = [self.path]
        if asset_group is not None:
            self._validate_asset_group(asset_group)
            path_parts.append(asset_group)
            if asset is not None:
                self._validate_asset(asset_group, asset)
                path_parts.append(asset)
                if sub_asset is not None:
                    self._validate_sub_asset(asset_group, asset, sub_asset)
                    path_parts.append(sub_asset)
                    if sub_2_asset is not None:
                        self._validate_sub_2_asset(asset_group, asset, sub_asset)
                        path_parts.append(sub_2_asset)
        element_path = "\\".join(path_parts)
        return element_path

    def _get_element_from_path(self, af_path: str):
        element = AFObject.FindObject(af_path)
        if element is None:
            raise Exception(f"No element found for specified path: {af_path}")
        return element

    def _get_elements(self, af_element=None) -> list:  # list of AFElement
        """Returns a list of AFElement objects (as opposed to an AFElements object)."""
        if af_element.Elements.Count == 0:
            return []
        return [e for e in af_element.Elements]

    def _get_element_names(self, af_element) -> list:
        """Returns a list of names for sub-elements of input element."""
        if af_element.Elements.Count == 0:
            return []
        return [e.Name for e in self._get_elements(af_element)]

    def _get_element_attributes(self, element):
        if element.Attributes.Count == 0:
            return []
        return [a.Name for a in element.Attributes]

    def get_attribute_names(self, asset_group=None, asset=None, sub_asset=None, sub_2_asset=None):
        # get element
        element_path = self._get_element_path(asset_group, asset, sub_asset, sub_2_asset)
        element = self._get_element_from_path(element_path)
        return self._get_element_attributes(element)

    def _validate_asset_args(self, asset_group: str, asset_names: list):
        self._validate_asset_group(asset_group)
        for asset in asset_names:
            self._validate_asset(asset_group, asset_name=asset)
        return

    def build_attribute_path_list(
        self,
        asset_group: str = "",
        asset_names: list = [],
        attributes: list = [],
    ):
        self._validate_asset_args(asset_group=asset_group, asset_names=asset_names)
        grp_path = self._get_element_path(asset_group=asset_group)
        specified = lambda attlist: [a for a in attributes if a in attlist]  # validates
        if len(asset_names) == 0:
            grp_atts = self.get_attribute_names(asset_group=asset_group)
            return [f"{grp_path}|{att}" for att in specified(grp_atts)]
        attribute_paths = []
        for asset in asset_names:
            asset_atts = self.get_attribute_names(asset_group=asset_group, asset=asset)
            attribute_paths.extend([f"{grp_path}\\{asset}|{att}" for att in specified(asset_atts)])
        return attribute_paths
