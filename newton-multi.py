import numpy as np

def newtons_method(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    """
    Multivariate Newton's method for optimizing f(x).

    Parameters
    ----------
    f       : objective function, f(x) -> scalar
    grad_f  : gradient function, grad_f(x) -> vector
    hess_f  : Hessian function, hess_f(x) -> matrix
    x0      : starting point (array-like)
    tol     : stopping tolerance on ||x_t - x_{t-1}||
    max_iter: maximum number of iterations

    Returns
    -------
    x       : the optimum found
    history : list of x values at each iteration
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for t in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)

        # Newton update: x_{t+1} = x_t - H^{-1} grad
        step = np.linalg.solve(hess, grad)  # avoids explicit inverse
        x_new = x - step

        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new

    return x, history


# Example usage: minimize f(x, y) = x^2 + y^2 (simple bowl, min at [0,0])
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

def hess_f(x):
    return np.array([[2, 0],
                      [0, 2]])

x_opt, hist = newtons_method(f, grad_f, hess_f, x0=[3.0, -2.0])
print("Optimum:", x_opt)
print("Iterations:", len(hist) - 1)