def derivative(f, x, epsilon=0.0001):
    """Estimate the first derivative of f at x using a forward difference.

    Keyword arguments:
    f -- the function to differentiate
    x -- the point at which to evaluate the derivative
    epsilon -- the step size used for the finite difference (default 0.0001)
    """
    return (f(x + epsilon) - f(x)) / epsilon


def second_derivative(f, x, epsilon=0.0001):
    """Estimate the second derivative of f at x using finite differences.

    Keyword arguments:
    f -- the function to differentiate
    x -- the point at which to evaluate the second derivative
    epsilon -- the step size used for the finite difference (default 0.0001)
    """
    return (
        derivative(f, x + epsilon, epsilon)
        - derivative(f, x, epsilon)
    ) / epsilon


def optimize(f, x0, tol=0.000001, epsilon=0.0001):
    """Find a stationary point of f using Newton's method.

    Starting from x0, repeatedly updates x using first and second
    derivative estimates until successive updates differ by less than
    tol, then returns the resulting point.

    Keyword arguments:
    f -- the function whose stationary point is sought
    x0 -- the starting point for the search
    tol -- the convergence tolerance; iteration stops when the change
           in x is smaller than this value (default 0.000001)
    epsilon -- the step size used for derivative estimates (default 0.0001)
    """
    x = x0
    while True:
        first_derivative = derivative(f, x, epsilon)
        second_deriv = second_derivative(f, x, epsilon)
        new_x = x - first_derivative / second_deriv
        if abs(new_x - x) < tol:
            return new_x
        x = new_x

