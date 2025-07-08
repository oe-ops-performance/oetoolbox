from functools import wraps


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
