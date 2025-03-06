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
