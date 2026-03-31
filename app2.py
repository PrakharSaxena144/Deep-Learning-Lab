import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("Neural Network Optimizer Visualization")

# ---------------- CONFIG ----------------
OPTIMIZERS = ["SGD", "Momentum", "RMSprop", "Adam", "Adagrad"]

COLORS = {
    "SGD": "#2563eb",
    "Momentum": "#7c3aed",
    "RMSprop": "#059669",
    "Adam": "#dc2626",
    "Adagrad": "#d97706",
}

# ---------------- DATA + OBJECTIVE ----------------
def make_synthetic_regression_data(n_samples, x_min, x_max, true_w1, true_w2, noise_std, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(x_min, x_max, n_samples)
    y = true_w1 * x + true_w2 + rng.normal(0.0, noise_std, size=n_samples)
    return x, y


def forward_pass(params, x):
    z1 = x[:, None] * params["w1"][None, :] + params["b1"][None, :]
    a1 = np.tanh(z1)
    y_hat = a1 @ params["w2"] + params["b2"]
    return y_hat, a1


def objective_fn(params, x, y):
    y_hat, _ = forward_pass(params, x)
    return float(np.mean((y_hat - y) ** 2))


def objective_grad(params, x, y):
    y_hat, a1 = forward_pass(params, x)
    err = y_hat - y
    n = x.shape[0]

    dy = (2.0 / n) * err
    gw2 = a1.T @ dy
    gb2 = float(np.sum(dy))

    da1 = dy[:, None] * params["w2"][None, :]
    dz1 = da1 * (1.0 - a1**2)
    gw1 = np.sum(dz1 * x[:, None], axis=0)
    gb1 = np.sum(dz1, axis=0)

    return {
        "w1": gw1,
        "b1": gb1,
        "w2": gw2,
        "b2": np.array(gb2, dtype=float),
    }


def init_params(w1, w2, hidden_neurons, seed):
    rng = np.random.default_rng(seed)
    return {
        "w1": np.array(w1 + rng.normal(0.0, 0.05, size=hidden_neurons), dtype=float),
        "b1": np.array(rng.normal(0.0, 0.05, size=hidden_neurons), dtype=float),
        "w2": np.array((w2 / max(hidden_neurons, 1)) + rng.normal(0.0, 0.05, size=hidden_neurons), dtype=float),
        "b2": np.array(float(w2), dtype=float),
    }


def proxy_linear_params(params, x):
    y_hat, _ = forward_pass(params, x)
    slope, intercept = np.polyfit(x, y_hat, 1)
    return float(slope), float(intercept)

# ---------------- OPTIMIZERS ----------------
def init_state(opt, params):
    zeros = {k: np.zeros_like(v) for k, v in params.items()}
    if opt == "Momentum":
        return {"v": zeros}
    if opt in ["RMSprop", "Adagrad"]:
        return {"s": zeros}
    if opt == "Adam":
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(val) for k, val in params.items()}
        return {"m": m, "v": v, "t": 1}
    return {}

def step(opt, params, grads, state, lr):
    eps = 1e-8
    beta = 0.9
    beta2 = 0.999

    p = {k: v.copy() for k, v in params.items()}

    if opt == "SGD":
        for k in p:
            p[k] -= lr * grads[k]

    elif opt == "Momentum":
        for k in p:
            state["v"][k] = beta * state["v"][k] - lr * grads[k]
            p[k] += state["v"][k]

    elif opt == "Adagrad":
        for k in p:
            state["s"][k] += grads[k]**2
            p[k] -= lr * grads[k] / (np.sqrt(state["s"][k]) + eps)

    elif opt == "RMSprop":
        for k in p:
            state["s"][k] = beta * state["s"][k] + (1-beta)*grads[k]**2
            p[k] -= lr * grads[k] / (np.sqrt(state["s"][k]) + eps)

    elif opt == "Adam":
        for k in p:
            state["m"][k] = beta * state["m"][k] + (1-beta)*grads[k]
            state["v"][k] = beta2 * state["v"][k] + (1-beta2)*(grads[k]**2)

            m_hat = state["m"][k] / (1 - beta**state["t"])
            v_hat = state["v"][k] / (1 - beta2**state["t"])

            p[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        state["t"] += 1

    return p, state

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

selected_opts = st.sidebar.multiselect("Select Optimizers", OPTIMIZERS, default=["Adam"])
show_all = st.sidebar.checkbox("Show All Optimizers")

st.sidebar.subheader("Initialize Starting Point")
init_w1 = st.sidebar.slider("Initial W1", -5.0, 5.0, -1.5)
init_w2 = st.sidebar.slider("Initial W2", -5.0, 5.0, -2.0)
epochs = st.sidebar.slider("Training Epochs", 50, 1000, 300, 50)
lr = st.sidebar.slider("Base Learning Rate", 0.001, 0.2, 0.01, 0.001)
hidden_neurons = st.sidebar.slider("Hidden Layer Neurons", 2, 10, 4, 1)

# Fixed synthetic dataset config (hidden from UI as requested)
n_samples = 80
x_min = -3.0
x_max = 3.0
true_w1 = 1.8
true_w2 = 0.3
noise_std = 0.35
seed = 42

x_data, y_data = make_synthetic_regression_data(
    n_samples=n_samples,
    x_min=x_min,
    x_max=x_max,
    true_w1=true_w1,
    true_w2=true_w2,
    noise_std=noise_std,
    seed=int(seed),
)

init_proxy_params = init_params(init_w1, init_w2, hidden_neurons, int(seed) + 1000)
init_proxy_w1, init_proxy_w2 = proxy_linear_params(init_proxy_params, x_data)

# ---------------- TRAIN ----------------
results = {}
adam_lr_factor = 0.5

for opt in OPTIMIZERS:
    params = init_params(init_w1, init_w2, hidden_neurons, int(seed) + 1000)

    state = init_state(opt, params)

    traj, losses = [], []
    status = "OK"
    opt_lr = lr * adam_lr_factor if opt == "Adam" else lr

    for e in range(epochs):
        proxy_w1, proxy_w2 = proxy_linear_params(params, x_data)
        l = objective_fn(params, x_data, y_data)

        if not np.isfinite(l):
            status = "Diverged"
            break

        traj.append({
            "x": proxy_w1,
            "y": proxy_w2,
            "w1": proxy_w1,
            "w2": proxy_w2,
            "loss": l,
            "epoch": e,
        })
        losses.append(l)

        grads = objective_grad(params, x_data, y_data)
        grads = {k: np.array(np.clip(v, -50, 50), dtype=float) for k, v in grads.items()}
        params, state = step(opt, params, grads, state, opt_lr)

    if status == "OK" and len(losses) > 1 and losses[-1] > losses[0] * 1.2:
        status = "Unstable"

    results[opt] = {"traj": traj, "losses": losses, "status": status, "lr": opt_lr}

# ---------------- SURFACE ----------------
def compute_surface():
    w1 = np.linspace(-5, 5, 80)
    w2 = np.linspace(-5, 5, 80)
    W1, W2 = np.meshgrid(w1, w2)
    # Vectorized MSE over the full parameter grid.
    pred = W1[..., None] * x_data + W2[..., None]
    Z = np.mean((pred - y_data) ** 2, axis=-1)

    return w1, w2, Z

wx, wy, Z = compute_surface()

# ---------------- FRAME ----------------
max_len = max(len(results[o]["traj"]) for o in OPTIMIZERS)
frame = st.slider("Step", 0, max_len-1, 0)

# ---------------- PLOT ----------------
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("MSE Surface (Proxy Slope/Intercept)", "Regression Fit (y vs x)"),
    horizontal_spacing=0.14,
)

fig.add_trace(
    go.Contour(
        x=wx,
        y=wy,
        z=Z,
        colorscale="RdBu",
        colorbar=dict(title="MSE", x=0.46, len=0.85),
        showscale=True,
        contours=dict(showlines=False),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=[init_proxy_w1],
        y=[init_proxy_w2],
        mode="markers",
        marker=dict(size=12, color="black"),
        name="Start",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=x_data,
        y=y_data,
        mode="markers",
        marker=dict(size=7, color="#1f2937", opacity=0.75),
        name="Synthetic data",
    ),
    row=1,
    col=2,
)

plot_opts = OPTIMIZERS if show_all else selected_opts

for opt in plot_opts:
    traj = results[opt]["traj"]

    xs = [p["x"] for p in traj]
    ys = [p["y"] for p in traj]

    show = min(frame+1, len(xs))

    fig.add_trace(
        go.Scatter(
            x=xs[:show],
            y=ys[:show],
            mode="lines+markers",
            name=opt,
            line=dict(color=COLORS[opt], width=3),
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )

    if show > 0:
        curr_w1 = xs[show - 1]
        curr_w2 = ys[show - 1]

        fig.add_trace(
            go.Scatter(
                x=[curr_w1],
                y=[curr_w2],
                mode="markers",
                name=f"{opt} (current)",
                marker=dict(size=11, color=COLORS[opt], symbol="circle-open", line=dict(width=2)),
                showlegend=False,
            )
            ,
            row=1,
            col=1,
        )

        y_fit = curr_w1 * x_data + curr_w2
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_fit,
                mode="lines",
                name=f"{opt} proxy fit",
                line=dict(color=COLORS[opt], width=3),
            ),
            row=1,
            col=2,
        )

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=20, r=20, t=60, b=20),
    height=520,
)

fig.update_xaxes(title_text="Proxy slope", row=1, col=1)
fig.update_yaxes(title_text="Proxy intercept", row=1, col=1)
fig.update_xaxes(title_text="x", row=1, col=2)
fig.update_yaxes(title_text="y", row=1, col=2)

st.plotly_chart(fig, use_container_width=True)

# ---------------- TABLE ----------------
st.subheader("📊 Weight Updates Table")

table_data = []

for opt in OPTIMIZERS:
    traj = results[opt]["traj"]
    if frame < len(traj):
        p = traj[frame]
    table_data.append([opt, p["epoch"], p["w1"], p["w2"], p["loss"]])

df = pd.DataFrame(table_data, columns=["Optimizer", "Epoch", "W1", "W2", "Loss"])
st.dataframe(df)

# ---------------- ALL OPTIMIZERS ON ONE GRAPH ----------------
st.subheader(f"MSE Curves - All Optimizers (up to Epoch {frame})")

fig2 = go.Figure()

for opt in OPTIMIZERS:
    losses = results[opt]["losses"]
    if not losses:
        continue

    upto = min(frame + 1, len(losses))
    xs = list(range(upto))
    ys = losses[:upto]

    fig2.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=opt,
            line=dict(color=COLORS[opt], width=3),
        )
    )

    point_idx = min(frame, len(losses) - 1)
    fig2.add_trace(
        go.Scatter(
            x=[point_idx],
            y=[losses[point_idx]],
            mode="markers",
            marker=dict(size=10, color=COLORS[opt]),
            showlegend=False,
            hovertemplate=f"{opt}<br>Epoch=%{{x}}<br>MSE=%{{y:.4f}}<extra></extra>",
        )
    )

fig2.update_layout(
    xaxis_title="Epoch",
    yaxis_title="MSE",
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- ADAM CHECK ----------------
adam_losses = results["Adam"]["losses"]
adam_status = results["Adam"]["status"]

if adam_losses:
    st.write(
        f"Adam health check: status={adam_status}, initial={adam_losses[0]:.4f}, final={adam_losses[-1]:.4f}, lr={results['Adam']['lr']:.4f}"
    )

if adam_status == "OK":
    st.success("Adam optimizer is stable for the selected setup.")
elif adam_status == "Unstable":
    st.warning("Adam appears unstable (final value notably higher than initial). Try lowering learning rate.")
else:
    st.error("Adam diverged (non-finite values encountered). Lower learning rate.")