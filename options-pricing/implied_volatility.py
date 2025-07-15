import math, numpy as np, pandas as pd, datetime as dt, yfinance as yf

ticker = "AAPL"
option_type = "call"
sigma_seed = 0.2

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

def bs_price_q(S, K, T, r, sigma, q=0.0, option="call"):
    if T <= 0:
        return max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
    if S <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option == "call":
        return S * math.exp(-q*T) * Phi(d1) - K * math.exp(-r*T) * Phi(d2)
    else:
        return K * math.exp(-r*T) * Phi(-d2) - S * math.exp(-q*T) * Phi(-d1)

def vega_q(S, K, T, r, sigma, q=0.0):
    if T <= 0 or S <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q*T) * phi(d1) * math.sqrt(T)

def implied_vol_newton(mkt, S, K, T, r, q=0.0, option="call", sigma_init=0.2, tol=1e-8, max_iter=50):
    lo, hi = 1e-6, 5.0
    sigma = sigma_init
    for _ in range(max_iter):
        f = bs_price_q(S, K, T, r, sigma, q, option) - mkt
        if abs(f) < tol:
            return sigma
        g = vega_q(S, K, T, r, sigma, q)
        if g <= 0 or math.isnan(g):
            break
        s_new = sigma - f / g
        if not (lo < s_new < hi):
            break
        sigma = s_new
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        fmid = bs_price_q(S, K, T, r, mid, q, option) - mkt
        if abs(fmid) < tol:
            return mid
        if fmid > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

tk = yf.Ticker(ticker)
S = tk.fast_info.get("lastPrice") or tk.history(period="1d")["Close"][-1]
div_yield = tk.info.get("dividendYield") or 0.0
q = float(div_yield) if div_yield else 0.0
irx = yf.Ticker("^IRX").fast_info.get("lastPrice")
if irx is None:
    irx = yf.Ticker("^IRX").history(period="1d")["Close"][-1]
r = float(irx) / 100.0

exps = tk.options
if not exps:
    raise SystemExit("No listed options found for this ticker.")
exp = exps[0]
chain = tk.option_chain(exp)
df = chain.calls if option_type == "call" else chain.puts

today = dt.datetime.utcnow()
expiry = dt.datetime.strptime(exp, "%Y-%m-%d")
T = max((expiry - today).days, 0) / 365.0

def mid_or_last(row):
    b, a = row.get("bid", np.nan), row.get("ask", np.nan)
    if np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0:
        return 0.5 * (b + a)
    lp = row.get("lastPrice", np.nan)
    return float(lp) if np.isfinite(lp) else np.nan

df = df.copy()
df["mid"] = df.apply(mid_or_last, axis=1)
df = df.dropna(subset=["mid"])

ivs = []
for _, row in df.iterrows():
    K = float(row["strike"])
    mkt = float(row["mid"])
    iv = implied_vol_newton(mkt, S, K, T, r, q=q, option=option_type, sigma_init=sigma_seed)
    ivs.append(iv)
df["impliedVol"] = ivs

print("Underlying:", ticker, "S=", round(S, 4), "r=", round(r, 6), "q=", round(q, 6), "exp:", exp, "T≈", round(T, 6))
print(df[["contractSymbol","strike","mid","impliedVol"]].head(10).to_string(index=False))