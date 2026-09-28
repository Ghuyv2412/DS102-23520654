import numpy as np

def f(x):
    x1 = x[0]
    x2 = x[1]

    return x1**4 + x2**2 + 2*x1*x2

def grad(f, x, h=1e-5):
    n = len(x)
    G = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        G[i] = (f(x_plus) - f(x_minus)) / (2*h)
    return G

def hess(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        H[:, i] = (grad(f, x_plus) - grad(f, x_minus)) / (2*h)
    return H

def classify_point(f, x):
    H = hess(f, x)
    eigenvalues = np.linalg.eigvalsh(H)

    if np.all(eigenvalues > 0):
        return "Cuc tieu"
    elif np.all(eigenvalues < 0):
        return "Cuc dai"
    else:
        return "Diem yen ngua"

def newton_raphson(f, x0, tol=1e-6, max_iter=100):

    x = np.array(x0, dtype=float)

    for k in range(max_iter):

        g = grad(f, x)

        if np.linalg.norm(g) < tol:
            break

        H = hess(f, x)

        try:
            delta_x = np.linalg.solve(H, -g)

        except np.linalg.LinAlgError:
            print("Hessian khong kha nghich.")
            return None

        x = x + delta_x

    return x


x0 = [1.0, -1.0]

result = newton_raphson(f, x0)

if result is not None:
    print("Diem tim duoc:", result)
    print("Gia tri ham f =", f(result))
    print("Loai diem:", classify_point(f, result))