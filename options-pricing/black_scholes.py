import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Utilities: normal CDF and analytic Black-Scholes ----------
def _phi(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_analytic_price(S, K, T, r, sigma, option="call"):
    if T <= 0:
        if option == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    if S <= 0:
        return 0.0 if option == "call" else K * math.exp(-r * T)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option == "call":
        return S * _phi(d1) - K * math.exp(-r * T) * _phi(d2)
    else:
        return K * math.exp(-r * T) * _phi(-d2) - S * _phi(-d1)

# ---------- Tridiagonal solver (Thomas algorithm) ----------
def thomas_tridiagonal(a, b, c, d):
    n = len(b)
    # Make copies to avoid in-place overwrite of inputs
    ac, bc, cc, dc = a.copy(), b.copy(), c.copy(), d.copy()

    # Forward sweep
    for i in range(1, n):
        w = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - w * cc[i - 1]
        dc[i] = dc[i] - w * dc[i - 1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x

# ---------- Black-Scholes PDE via Crank-Nicolson ----------
def black_scholes_pde_cn(
    S_max=400.0,
    M=400,              # number of price steps (S grid)
    T=1.0,              # years to maturity
    N=1000,             # number of time steps
    r=0.05,
    sigma=0.2,
    K=100.0,
    option="call",      # "call" or "put"
    S0=100.0,           # spot for diagnostic
    plot=True
):
    # Grid
    dS = S_max / M
    dt = T / N
    S = np.linspace(0.0, S_max, M + 1)  # i=0..M

    # Terminal condition (payoff at maturity)
    if option == "call":
        V = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)

    # Pre-allocate arrays for tridiagonal system (interior points 1..M-1)
    i = np.arange(1, M)  # interior indices
    # Coefficients for CN in i-space with S_i = i*dS (standard discretization)
    # Following the common CN formulation (e.g., Wilmott):
    # Define helpers based on i (since S_i = i*dS => S_i/dS = i)
    i2 = i.astype(float) ** 2

    # These are the "half-step" coefficients for the operator L:
    # L V_i ≈ 0.5*sigma^2*i^2*(V_{i+1} - 2V_i + V_{i-1})
    #         + 0.5*r*i*(V_{i+1} - V_{i-1}) - r*V_i
    # CN uses (I - 0.5*dt*L) V^{n} = (I + 0.5*dt*L) V^{n+1}
    alpha = 0.25 * dt * (sigma**2 * i2 - r * i)     # sub-diagonal contribution
    beta  = -0.5 * dt * (sigma**2 * i2 + r)         # diagonal contribution
    gamma = 0.25 * dt * (sigma**2 * i2 + r * i)     # super-diagonal contribution

    # LHS matrix diagonals (for V^n unknown):  -alpha, 1 - beta, -gamma
    a_lhs = -alpha[1:]                 # length M-2 (subdiag)
    b_lhs = 1.0 - beta                 # length M-1 (diag)
    c_lhs = -gamma[:-1]                # length M-2 (superdiag)

    # Time-marching backward: from n=N-1 ... 0
    for n in range(N, 0, -1):
        t = (n - 1) * dt  # time after stepping (we move toward t=0)

        # Right-hand side using V^{n} (currently V holds V^{n})
        # RHS_i = alpha*V_{i-1} + (1 + beta)*V_i + gamma*V_{i+1}
        rhs = alpha * V[i - 1] + (1.0 + beta) * V[i] + gamma * V[i + 1]

        # Apply boundary conditions in RHS (Dirichlet BCs)
        if option == "call":
            V_0_t   = 0.0
            V_SM_t  = S_max - K * math.exp(-r * (T - t))
        else:
            V_0_t   = K * math.exp(-r * (T - t))
            V_SM_t  = 0.0

        # Adjust the first and last entries of RHS due to boundaries
        rhs[0]      -= (-alpha[0]) * V_0_t        # because a_lhs affects i=1 via V_0
        rhs[-1]     -= (-gamma[-1]) * V_SM_t      # because c_lhs affects i=M-1 via V_M

        # Solve tridiagonal for interior nodes
        V_interior = thomas_tridiagonal(a_lhs, b_lhs, c_lhs, rhs)

        # Update solution including boundaries
        V[0]   = V_0_t
        V[1:M] = V_interior
        V[M]   = V_SM_t

    # Diagnostics against analytic price at S0 (nearest grid node)
    idx_S0 = int(np.argmin(np.abs(S - S0)))
    V_grid = float(V[idx_S0])
    V_bs   = float(bs_analytic_price(S0, K, T, r, sigma, option))
    abs_err = abs(V_grid - V_bs)

    print("=== Black–Scholes PDE (Crank–Nicolson) ===")
    print(f"Option: {option.upper()} | K={K} | r={r} | sigma={sigma} | T={T}")
    print(f"Grid: M={M} (dS={dS:.4f}), N={N} (dt={dt:.6f}), S_max={S_max}")
    print(f"S0≈{S[idx_S0]:.4f}  PDE={V_grid:.6f}  Analytic={V_bs:.6f}  |abs err|={abs_err:.6e}")

    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(S, V, linewidth=2, label="PDE (t=0)")
        plt.axvline(S[idx_S0], linestyle="--", linewidth=1, label=f"S0≈{S[idx_S0]:.2f}")
        plt.title(f"European {option.capitalize()} Price via PDE (t=0)")
        plt.xlabel("Underlying price S")
        plt.ylabel("Option value V(S,0)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return S, V, V_grid, V_bs, abs_err

# ---------- Example run ----------
if __name__ == "__main__":
    # You can tweak these to test convergence/smoothness, like Tianyi suggested.
    S, V, V_grid, V_bs, err = black_scholes_pde_cn(
        S_max=400.0,
        M=400,
        T=1.0,
        N=1200,          # CN is unconditionally stable; larger N improves accuracy
        r=0.05,
        sigma=0.2,
        K=100.0,
        option="call",
        S0=100.0,
        plot=True
    )
