def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Return final x after 'steps' iterations.
    """

    x = x0

    for step in range(0, steps, 1):
        grad = 2 * a * x + b
        x = x - lr * grad
    return float(x)