import clr, sys

sys.path.append(r"C:\Program Files (x86)\PIPC\AF\PublicAssemblies\4.0")
clr.AddReference("OSIsoft.AFSDK")
import OSIsoft  # type: ignore
from OSIsoft.AF import AFObject, PISystems  # type: ignore

pisystem = PISystems().DefaultPISystem
database = pisystem.Databases.get_Item("Onward Energy")


class PISiteElement:
    """A class for accessing PI Element metadata via AFSDK."""

    def __init__(self, fleet: str, site_name: str, q: bool = True):

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
        if not q:
            print("Loading AF structure from PI", end=" ... ")
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

        if not q:
            print("done.")
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
                    path_parts.append(asset)
                    if sub_2_asset is not None:
                        self._validate_sub_2_asset(asset_group, asset, sub_asset)
                        path_parts.append(asset)
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
