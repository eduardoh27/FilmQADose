"""
functions.py

This module defines a FittingFunction class that encapsulates a fitting function,
its name, descriptive text, the names of its fitting parameters, and the possible types
of independent variables ("x") that can be used (e.g., netOD, netT, reflectance).
"""

class FittingFunction:
    def __init__(self, name: str, func, initial_param_guess: list, description: str, param_names: list, independent_variable: str):
        """
        Initializes a FittingFunction instance.

        Parameters
        ----------
        name : str
            The name of the fitting function.
        func : callable
            The fitting function. The independent variable (x) should be its first argument.
        description : str
            A descriptive text (e.g., a LaTeX string) representing the fitting function.
        param_names : list
            A list of fitting parameter names (e.g., ["a", "b", "n"]).
        x_types : list
            The independent variable of the function 
            (e.g., "netOD", "netT").
        """
        self.name = name
        self.func = func
        self.initial_param_guess = initial_param_guess
        self.description = description
        self.param_names = param_names
        self.independent_variable = independent_variable

    def __repr__(self):
        return (f"FittingFunction(name={self.name}, "
                f"param_names={self.param_names}, x_types={self.x_types})")

# Define the polynomial fitting function.
def polynomial(x, a, b, n):
    """
    Polynomial fitting function.
    
    Parameters
    ----------
    x : float or array-like
        The independent variable (e.g., netOD).
    a, b, n : float
        Fitting parameters.
    
    Returns
    -------
    float or array-like
        Calculated dose.
    """
    return a * x + b * (x ** n)

# Define the rational fitting function.
def rational(x, a, b):
    """
    Rational fitting function.
    
    Parameters
    ----------
    x : float or array-like
        The independent variable (e.g., netOD).
    a, b : float
        Fitting parameters.
    
    Returns
    -------
    float or array-like
        Calculated dose.
    """
    return (a * x) / (1 - b * x)

# Create FittingFunction instances for each fitting method.
# You can define the supported x_types based on your application needs.
polynomial_fitting = FittingFunction(
    name="polynomial",
    func=polynomial,
    initial_param_guess=[1.0, 1.0, 1.0],
    description=r"$D = a\,netOD + b\,netOD^n$",
    param_names=["a", "b", "n"],
    independent_variable = "netOD"
)

rational_fitting = FittingFunction(
    name="rational",
    func=rational,
    initial_param_guess=[2.0, 2.0],
    description=r"$D = \frac{a\,netT}{1 - b\,netT}$",
    param_names=["a", "b"],
    independent_variable= "netT"
)

def cuadratic(x, a, b, c):
    return a * x**2 + b * x + c 

cuadratic_fitting = FittingFunction(
    name="cuadratic",
    func=cuadratic,
    initial_param_guess=[1.0, 1.0, 1.0],
    description=r"$D = a\,netOD^2 + b\,netOD + c$",
    param_names=["a", "b", "c"],
    independent_variable= "netOD"
)


# Dictionary mapping function names to their corresponding FittingFunction instances.
fitting_functions = {
    "polynomial": polynomial_fitting,
    "rational": rational_fitting,
    "cuadratic": cuadratic_fitting
}

def get_fitting_function(name: str) -> FittingFunction:
    """
    Retrieves the FittingFunction instance by name.
    
    Parameters
    ----------
    name : str
        The name of the fitting function (e.g., "polynomial").
    
    Returns
    -------
    FittingFunction
        The corresponding FittingFunction instance if found; otherwise, None.
    """
    return fitting_functions.get(name)
