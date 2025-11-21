from functools import wraps
import inspect


def with_retries(n_max: int = 5, raise_error: bool = True):
    """Decorator factory - returns retry decorator and supports additional args.
    -> to be applied at function definition
    -> if keyword argument "q=True" exists in target function, enables retry status printouts
    """

    def decorator_retries(func):
        """Decorator for retry wrapper"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper for running a function with retry logic"""
            printouts = False
            if "q" in kwargs:
                printouts = kwargs.get("q") is False
            qprint = lambda msg: None if not printouts else print(msg)
            qprint(f"< Implementing retry logic for function: {func.__name__} >")
            error_messages = []
            while len(error_messages) < n_max:
                try:
                    output = func(*args, **kwargs)
                    qprint(f"< Function completed after {len(error_messages) + 1} attempts. >")
                    return output
                except Exception as e:
                    error_messages.append(str(e))
            if not raise_error:
                return
            messages = [f"ERROR {i+1}: {e}" for i, e in enumerate(error_messages)]
            message_str = "\n".join(messages)
            raise Exception(f"Failed after {n_max} retries.\n{message_str}")

        return wrapper

    return decorator_retries


def quiet_print_function(q: bool):
    """wrapper for print function to be used in functions with quiet parameter

    Parameters
    ----------
    q : bool
        The parameter that determines whether to print

    Returns
    -------
    function
        A print function that only prints when quiet parameter is False
    """
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)
    return qprint


def print_dataframe_info(df):
    print(f"<class '{type(df)}'>")
    idx_type = str(type(df.index)).split(".")[-1]
    idx_min, idx_max = str(df.index.min()), str(df.index.max())
    print(f"{idx_type}: {len(df.index)} entries, {idx_min} to {idx_max}")

    print(f"Data columns (total {df.shape[1]} columns):")

    max_non_null_str = str(df.notna().sum().max()) + " non-null"
    include_non_null = len(max_non_null_str) <= len("Non-Null Count")
    p2_ljust = max(len(str(c)) for c in df.columns[:10])
    hdr_1a = " # "
    hdr_1b = "---"
    hdr_2a = "Column".ljust(p2_ljust)
    hdr_2b = "------".ljust(p2_ljust)
    hdr_3a, hdr_3b = "", ""
    if include_non_null:
        hdr_3a = "Non-Null Count"
        hdr_3b = "-" * len(hdr_3a)
    hdr_4a = "Dtype"
    hdr_4b = "-----"
    print("  ".join([hdr_1a, hdr_2a, hdr_3a, hdr_4a]))
    print("  ".join([hdr_1b, hdr_2b, hdr_3b, hdr_4b]))

    # only printing up to a maximum of 10 columns
    limit = 10
    for i, item in enumerate(df.notna().sum().to_dict().items()):
        if limit == 0:
            break
        col, n_non_null = item
        part_1 = f" {i}".ljust(3)
        part_2 = col.ljust(p2_ljust)
        part_3 = ""
        if include_non_null:
            part_3 = f"{n_non_null} non-null".ljust(len(hdr_3a))
        part_4 = str(df.dtypes[col])
        print("  ".join([part_1, part_2, part_3, part_4]))
        limit -= 1

    if df.shape[1] > 10:
        print("...")

    return


def print_raw_module_info(module):
    for name, obj in inspect.getmembers(module):
        if not name.startswith("__"):
            print(f"{name}: {obj}")
    return


def list_module_functions(module):
    """Lists all functions explicitly defined within a given module."""
    functions = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and inspect.getmodule(obj) is module:
            functions.append(name)
    return functions


def members_by_type(cls):
    prop_dict = {}
    for name, member in inspect.getmembers(cls):
        if not name.startswith("__"):  # Exclude special attributes
            type_str = str(type(member)).replace("<class '", "").replace("'>", "")
            if type_str not in prop_dict:
                prop_dict[type_str] = [name]
            else:
                prop_dict[type_str].append(name)
    return prop_dict


def print_class_information(cls):
    print(f"Information for class '{cls.__name__}'")
    prop_dict = members_by_type(cls)
    types_ = list(sorted([*prop_dict], reverse=True))
    for type_str in types_:
        print("")
        member_list = prop_dict[type_str]
        members = list(sorted([x for x in member_list if not x.startswith("_")]))
        for name in members:
            print(f"  {type_str} - {name}")
    return


def get_filtered_members(cls, obj_type: str):
    """function for inspecting properties and functions of a class"""
    if obj_type not in ["property", "function"]:
        raise ValueError("Invalid obj_type specified. Must be 'property' or 'function'.")
    filtered_members = []
    for name, member in inspect.getmembers(cls):
        if name.startswith("__"):  # Exclude special attributes
            continue
        if obj_type in str(type(member)):
            filtered_members.append((name, member))
    return filtered_members


def get_member_names(cls, obj_type: str):
    return [x[0] for x in get_filtered_members(cls, obj_type)]


def print_class_info(cls, member_type=None):
    class_str = ".".join([cls.__module__, cls.__name__])
    print("{{ " + class_str + " }}")
    member_type_list = ["property", "function"]
    if member_type is not None:
        if member_type not in member_type_list:
            print("Invalid member type.")
            return
        member_type_list = [member_type]
    for mtype in member_type_list:
        print(f"[{mtype}]")
        obj_list = get_filtered_members(cls, mtype)
        for name, obj in obj_list:
            if name.startswith("_"):
                continue
            func = obj if mtype == "function" else obj.fget
            arg_str = str(inspect.signature(func))
            obj_str = func.__name__ + arg_str
            print(f"    .{obj_str}")
    return
