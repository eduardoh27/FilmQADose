"""
functions.py

This module defines various fitting functions along with their descriptive texts.
Each function is identified by a name (as a string) and can be retrieved when needed.
"""

def exponential(netOD, a, b, n):
    """
    Exponential fitting function.
    
    Parameters
    ----------
    netOD : float or array-like
        Net optical density.
    a, b, n : float
        fitting parameters.
    
    Returns
    -------
    float or array-like
        Calculated dose.
    """
    return a * netOD + b * (netOD ** n)

# Dictionary mapping function names to their corresponding fitting functions.
fitting_functions = {
    "exponential": exponential,
}

# Dictionary mapping function names to their descriptive text.
fitting_function_texts = {
    "exponential": r"$D = a netOD + b netOD^n$",
}

def get_fitting_function(name):
    """
    Retrieves the fitting function and its description text by name.
    
    Parameters
    ----------
    name : str
        The name of the fitting function (e.g., "exponential").
    
    Returns
    -------
    tuple
        A tuple (function, description_text) if the function exists; otherwise, (None, None).
    """
    func = fitting_functions.get(name)
    text = fitting_function_texts.get(name, "")
    return func, text
