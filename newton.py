def derivative(f, x, epsilon=0.0001):
    return (f(x + epsilon) - f(x)) / epsilon


def second_derivative(f, x, epsilon=0.0001):
    return (
        derivative(f, x + epsilon, epsilon)
        - derivative(f, x, epsilon)
    ) / epsilon


def newton(f, x0, tol=0.000001, epsilon=0.0001):
    x = x0

    while True:
        first_derivative = derivative(f, x, epsilon)
        second_deriv = second_derivative(f, x, epsilon)

        new_x = x - first_derivative / second_deriv

        if abs(new_x - x) < tol:
            return new_x

        x = new_x

import newton
import numpy as np
newton.optimize(start, fun)   ## Assuming your function is called `optimize`.
newton.optimize(2.5, np.cos)  ## Minimizing cos(x) from close-ish to one minimum.