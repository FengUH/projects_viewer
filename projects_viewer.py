import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ==========================
# Page configuration
# ==========================
st.set_page_config(
    page_title="Project Viewer - Feng",
    page_icon="🧠",
    layout="wide",
)

# Header
st.markdown(
    """
    <div style="padding: 0.5rem 0 0.3rem 0;">
      <h1 style="margin-bottom: 0.2rem;">🧠 Feng Neuroscience Research Tool</h1>
      <p style="color:#6b7280; margin-bottom:0.5rem; line-height:1.45;">
        A neuroscience research tool developed by Feng — integrating single-subject EEG <b>time-series processing</b>, 
        group-level <b>statistical and regression analysis</b>, <b>machine-learning</b>–based disease classification, 
        and <b>k-means</b>–driven dynamic brain-state discovery.
      </p>
    </div>
    <hr style="margin-top:0.3rem; margin-bottom:0.8rem;"/>
    """,
    unsafe_allow_html=True,
)

# Reusable HTML divider
LIGHT_DIVIDER = """
<hr style="
  margin: 1.0rem 0 1.0rem 0;
  border: 0;
  border-top: 2px solid #d4d4d8;
  background-color: #e5e7eb;
"/>
"""

# ==========================
# Simulated EEG (synthetic demo data)
# ==========================
@st.cache_data
def simulate_eeg(n_channels=16, n_seconds=10, fs=256):
    t = np.arange(0, n_seconds, 1.0 / fs)
    data = np.zeros((n_channels, t.size))
    rng = np.random.default_rng(42)

    for ch in range(n_channels):
        delta = np.sin(2 * np.pi * rng.uniform(1, 3) * t)
        alpha = 0.5 * np.sin(
            2 * np.pi * rng.uniform(8, 12) * t + rng.uniform(0, 2 * np.pi)
        )
        beta = 0.3 * np.sin(
            2 * np.pi * rng.uniform(15, 25) * t + rng.uniform(0, 2 * np.pi)
        )
        noise = 0.5 * rng.normal(size=t.size)
        data[ch, :] = delta + alpha + beta + noise

    channel_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "Fz",
        "Cz",
        "Pz",
        "T3",
        "T4",
        "Oz",
    ][:n_channels]

    return t, data, channel_names


FS = 256
N_CHANNELS = 16
N_SECONDS = 10
t, eeg_data, channel_names = simulate_eeg(N_CHANNELS, N_SECONDS, FS)

# ==========================
# Session state for mock loading & ML workflow
# ==========================
for key, default in [
    ("single_loaded", False),
    ("group_loaded", False),
    ("model_trained", False),
    ("model_selected", False),
    ("model_applied", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================
# Sidebar: analysis workflow controller
# ==========================
st.sidebar.markdown("### 🧭 Analysis mode")

analysis_mode = st.sidebar.radio(
    "Select analysis mode",
    ["Single-subject analysis", "Group analysis"],
)

# Defaults for shared controls
window_length = 2.0
window_start = 1.0
n_show_channels = 8
group_task = "None"  # will be updated in group branch

# -------- Single-subject branch --------
if analysis_mode == "Single-subject analysis":
    st.sidebar.markdown("#### Single-subject analysis")

    load_single_btn = st.sidebar.button(
        "📂 Load single-subject", key="btn_single_load"
    )
    if load_single_btn:
        st.session_state["single_loaded"] = True

    if st.session_state["single_loaded"]:
        st.sidebar.success("Single-subject file loaded.")

        st.sidebar.markdown("---")
        st.sidebar.markdown("##### Time window & channels")

        window_length = st.sidebar.number_input(
            "Window length (s)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
        )

        max_start = max(0.0, N_SECONDS - window_length)
        window_start = st.sidebar.number_input(
            "Window start (s)",
            min_value=0.0,
            max_value=float(max_start),
            value=1.0,
            step=0.1,
        )

        n_show_channels = st.sidebar.number_input(
            "Channels to show",
            min_value=1,
            max_value=N_CHANNELS,
            value=8,
            step=1,
        )
    # else:
    #     st.sidebar.info("Click the button above to mock-load a single-subject dataset.")

# -------- Group branch --------
elif analysis_mode == "Group analysis":
    st.sidebar.markdown("#### Group analysis")

    load_group_btn = st.sidebar.button(
        "📂 Load group-subjects", key="btn_group_load"
    )
    if load_group_btn:
        st.session_state["group_loaded"] = True

    if st.session_state["group_loaded"]:
        st.sidebar.success("Group-level subjects loaded.")

        group_task = st.sidebar.radio(
            "Select analysis type",
            ["None", "Statistics", "ML classification"],
            index=0,
        )

        if group_task == "ML classification":
            # ---- ML workflow in a nicer expander ----
            with st.sidebar.expander("ML workflow", expanded=True):
                st.markdown(
                    "<div style='font-size:0.85rem; color:#6b7280; margin-bottom:0.4rem;'>"
                    "Step 1: choose model<br/> " \
                    "Step 2: Diagnosis of disease"
                    "</div>",
                    unsafe_allow_html=True,
                )

                # ----- Sub-section 1: Select model -----
                st.markdown("**① Choose model**")

                train_btn = st.button(
                    "⚙️ Train model on group",
                    key="btn_train_model",
                    help="Simulate training a CNN on group-level connectivity features.",
                )
                if train_btn:
                    st.session_state["model_trained"] = True
                    st.session_state["model_selected"] = True

                existing_btn = st.button(
                    "📂 Load existing model",
                    key="btn_existing_model",
                    help="Mock-select a pre-trained model from disk.",
                )
                if existing_btn:
                    st.session_state["model_selected"] = True
                    st.session_state["model_trained"] = False

                if st.session_state.get("model_trained"):
                    st.success("Trained model ready.")
                elif st.session_state.get("model_selected"):
                    st.info("Existing model selected.")

                st.markdown("---")

                # ----- Sub-section 3: Demo single-subject prediction -----
                st.markdown("**② Diagnosis of disease**")

                load_subject_btn = st.button(
                    "📂 Load one subject",
                    key="btn_load_demo_subject",
                    help="Mock-import a single subject's connectivity matrix.",
                )
                if load_subject_btn:
                    st.session_state["demo_subject_loaded"] = True
                    st.info("Subject loaded.")

                apply_subject_btn = st.button(
                    "🔍 Apply model to subject",
                    key="btn_apply_subject",
                    help="Run the CNN on the demo subject and return a diagnosis.",
                )
                if apply_subject_btn:
                    if not (st.session_state.get("model_trained") or st.session_state.get("model_selected")):
                        st.warning("Train or select a model first.")
                    elif not st.session_state.get("demo_subject_loaded"):
                        st.warning("Load a demo subject first.")
                    else:
                        rng = np.random.default_rng(2025)
                        pred_label = rng.choice(["Depression", "AD"])
                        st.session_state["subject_prediction"] = pred_label
                        st.success(f"Diagnostic result for the subject: {pred_label}")

    # else:
    #     st.sidebar.info("Click the button above to load multiple subjects.")

# ==========================
# Utility: moving-average smoothing
# ==========================
def moving_average_filter(data, kernel_size=15):
    """Apply a 1D moving-average filter channel-wise."""
    if kernel_size < 3:
        return data
    kernel = np.ones(kernel_size) / kernel_size
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch, :] = np.convolve(data[ch, :], kernel, mode="same")
    return filtered


# ==========================
# Utility: multi-channel EEG plotting
# ==========================
def plot_eeg_signals(t_local, data, channel_names_local, title):
    """
    Plot multi-channel EEG with automatic vertical spacing and padded
    y-limits for a cleaner presentation.
    """
    n_channels_local = data.shape[0]

    stds = np.std(data, axis=1)
    spacing = np.max(stds) * 4.0
    pad = spacing * 0.6

    fig, ax = plt.subplots(figsize=(10, 3))

    for idx in range(n_channels_local):
        offset = idx * spacing
        ax.plot(t_local, data[idx, :] + offset, linewidth=0.9)
        ax.text(
            t_local[0] - (t_local[-1] * 0.015),
            offset,
            channel_names_local[idx],
            va="center",
            ha="right",
            fontsize=8,
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.grid(alpha=0.22, linestyle="--")

    ymin = -pad
    ymax = (n_channels_local - 1) * spacing + pad
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    return fig


# ==========================
# Utility: compact connectivity matrix plot
# ==========================
CMAP = "viridis"


def plot_connectivity_matrix(
    conn_matrix,
    chan_names,
    title="Connectivity Matrix",
    vmin=-1,
    vmax=1,
    fig_size=(3.0, 1.8),
    title_fontsize=8,
    tick_fontsize=5,
    cbar_labelsize=6,
):
    """Render a correlation-based connectivity matrix with minimal margins."""
    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.imshow(conn_matrix, vmin=vmin, vmax=vmax, origin="lower", cmap=CMAP)

    ax.set_xticks(range(len(chan_names)))
    ax.set_yticks(range(len(chan_names)))
    ax.set_xticklabels(chan_names, rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(chan_names, fontsize=tick_fontsize)

    if title:
        ax.set_title(title, fontsize=title_fontsize)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.03)
    cbar.ax.tick_params(labelsize=tick_fontsize - 1)
    cbar.set_label("Corr", fontsize=cbar_labelsize)

    plt.subplots_adjust(left=0.22, right=0.86, top=0.82, bottom=0.32)
    return fig


# ==========================
# Extract current time window and backend connectivity
# ==========================
start_idx = int(window_start * FS)
end_idx = int((window_start + window_length) * FS)
start_idx = max(0, start_idx)
end_idx = min(eeg_data.shape[1], end_idx)

t_window = t[start_idx:end_idx]
eeg_window = eeg_data[: int(n_show_channels), start_idx:end_idx]
chan_window_names = channel_names[: int(n_show_channels)]

# Backend connectivity for group analysis (fixed smoothing)
eeg_filtered_backend = moving_average_filter(eeg_data, kernel_size=3)
eeg_filtered_window_backend = eeg_filtered_backend[
    : int(n_show_channels), start_idx:end_idx
]
conn_matrix_backend = np.corrcoef(eeg_filtered_window_backend)

# ======================================================
# Tabs
# ======================================================
tab1, tab2, tab3 = st.tabs(
    [
        "1. Single-subject EEG time series",
        "2. Statistics & Regression",
        "3. ML-based Classification",
    ]
)

# ------------------------------------------------------
# Tab 1: EEG signals + 3 parallel analysis options
# ------------------------------------------------------
with tab1:
    if analysis_mode == "Single-subject analysis" and st.session_state["single_loaded"]:
        st.markdown("#### Single-subject EEG time series")

        col_left, _ = st.columns([3, 2])
        with col_left:
            st.caption(
                f"Showing {int(n_show_channels)} channels, "
                f"time window {window_start:.1f}–{window_start+window_length:.1f} s."
            )

        # Raw EEG (always shown)
        fig_raw = plot_eeg_signals(
            t_window,
            eeg_window,
            chan_window_names,
            title="Raw EEG",
        )
        st.pyplot(fig_raw, clear_figure=True)

        # ---------- Analysis options block ----------
        st.markdown(LIGHT_DIVIDER, unsafe_allow_html=True)
        st.markdown("##### Advanced EEG Analysis")

        # 事先定义，供后续选项复用
        eeg_window_filtered_local = None

        # ==================================================
        # 1) Smoothing option + results directly underneath
        # ==================================================
        st.markdown("###### 1) Apply moving-average smoothing")
        opt_smooth = st.checkbox(
            "Show smoothed EEG (moving-average filter)",
            value=False,
        )

        kernel_local = 3
        if opt_smooth:
            # 让输入框处在一个比较窄的列里，使其宽度接近“Kernel size”文字
            col_k, col_help = st.columns([1, 4])
            with col_k:
                st.markdown(
                    "<div style='font-size:0.9rem; font-weight:500; "
                    "margin-bottom:0.1rem;'>Kernel size</div>",
                    unsafe_allow_html=True,
                )
                kernel_local = st.number_input(
                    "",
                    min_value=3,
                    max_value=51,
                    step=2,
                    value=3,
                    help="Length of the moving-average window (odd number).",
                    label_visibility="collapsed",
                )
            with col_help:
                st.caption("Applied channel-wise in the time domain.")

            # ----- Smoothed EEG -----
            eeg_window_filtered_local = moving_average_filter(
                eeg_window, kernel_size=int(kernel_local)
            )

            fig_filt = plot_eeg_signals(
                t_window,
                eeg_window_filtered_local,
                chan_window_names,
                title=f"Smoothed EEG (kernel = {int(kernel_local)})",
            )
            st.pyplot(fig_filt, clear_figure=True)

            st.markdown(
                f"<div style='color:#6b7280; font-size:0.85rem; text-align:center; "
                f"margin-top:0.3rem;'>"
                f"Centered moving-average smoothing with window size "
                f"<code>K = {int(kernel_local)}</code>."
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='text-align: center; margin-top:0.2rem;'>", unsafe_allow_html=True)
            st.latex(
                r"\tilde{x}(t) = \frac{1}{K}\sum_{i=-\frac{K-1}{2}}^{\frac{K-1}{2}} x(t + i)"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # 分隔线，让三个分析模块视觉上独立
        st.markdown(LIGHT_DIVIDER, unsafe_allow_html=True)

        # ==================================================
        # 2) Connectivity option + results directly underneath
        # ==================================================
        st.markdown("###### 2) Show window-level connectivity")
        opt_conn = st.checkbox(
            "Show connectivity matrix (current window)",
            value=False,
        )

        if opt_conn:
            # use filtered data if available, otherwise raw
            if eeg_window_filtered_local is not None:
                conn_matrix_local = np.corrcoef(eeg_window_filtered_local)
            else:
                conn_matrix_local = np.corrcoef(eeg_window)

            # 标题 + 简短描述
            st.markdown(
                """
                <div style='font-weight:600; margin-top:0.4rem; margin-bottom:0.15rem;'>
                  Window-level connectivity matrix
                </div>
                <div style='color:#6b7280; font-size:0.85rem; margin-bottom:0.2rem;'>
                  Symmetric correlation matrix <em>R</em> summarizing linear
                  dependencies among channels within this time window.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --- Plot matrix ---
            fig_conn = plot_connectivity_matrix(
                conn_matrix_local,
                chan_window_names,
                title="",
                vmin=-1,
                vmax=1,
                fig_size=(6, 3.8),
                title_fontsize=8,
                tick_fontsize=6,
                cbar_labelsize=9,
            )

            col_l, col_c, col_r = st.columns([1, 2, 1])
            with col_c:
                st.pyplot(fig_conn, clear_figure=True, use_container_width=False)

            # 公式紧跟在矩阵下方
            st.markdown(
                "<div style='text-align:center; margin-top:0.5rem; margin-bottom:0.35rem;'>",
                unsafe_allow_html=True,
            )
            st.latex(
                r"\rho_{ij} = \frac{\mathrm{cov}(X_i, X_j)}{\sigma_i \sigma_j},"
                r"\qquad R = [\rho_{ij}]_{i,j=1}^{N_{\text{chan}}}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # 灰色小字 caption
            st.markdown(
                """
                <div style='color:#6b7280; font-size:0.80rem; 
                            text-align:center; margin-top:0.10rem; line-height:1.4;'>
                    Correlation matrix <em>R</em> for this window (Pearson coefficients 
                    ρ<sub>ij</sub> between channels i and j).
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 再加一条分隔线，和 dFNC 模块隔开
        st.markdown(LIGHT_DIVIDER, unsafe_allow_html=True)

        # ==================================================
        # 3) k-means dFNC option + results directly underneath
        # ==================================================
        st.markdown("###### 3) Identify dynamic connectivity states")
        opt_dfnc = st.checkbox(
            "Identify dynamic brain states (k-means dFNC)",
            value=False,
        )

        if opt_dfnc:

            def compute_dynamic_conn_and_states(
                eeg, fs, win_len_sec=2.0, step_ratio=0.5, n_states=4, n_iter=30
            ):
                """
                Compute sliding-window functional connectivity and cluster windows into
                a finite set of connectivity states using k-means.
                """
                rng_km = np.random.default_rng(2025)
                n_channels_local, T_local = eeg.shape
                win_len_local = int(win_len_sec * fs)
                step_local = max(1, int(win_len_local * step_ratio))

                if win_len_local <= 1 or win_len_local > T_local:
                    win_len_local = max(2, min(T_local, win_len_local))

                starts_local = np.arange(0, T_local - win_len_local + 1, step_local)
                if len(starts_local) == 0:
                    starts_local = np.array([0])
                    win_len_local = T_local

                conn_list_local = []
                t_centers_local = []

                tri_idx_local = np.triu_indices(n_channels_local, k=1)
                feat_list_local = []

                for s_local in starts_local:
                    e_local = s_local + win_len_local
                    seg = eeg[:, s_local:e_local]
                    c = np.corrcoef(seg)
                    conn_list_local.append(c)
                    feat_list_local.append(c[tri_idx_local])
                    center_t = (s_local + e_local - 1) / 2.0 / fs
                    t_centers_local.append(center_t)

                conn_list_local = np.stack(conn_list_local, axis=0)
                feats_local = np.stack(feat_list_local, axis=0)
                t_centers_local = np.array(t_centers_local)

                def kmeans_nd(X, k, n_iter=30, rng=None):
                    """Minimal k-means implementation for high-dimensional feature vectors."""
                    if rng is None:
                        rng = np.random.default_rng(2025)
                    n_samples = X.shape[0]
                    k = min(k, n_samples)
                    idx0 = rng.choice(n_samples, size=k, replace=False)
                    centers = X[idx0].copy()
                    for _ in range(n_iter):
                        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                        labels = np.argmin(dists, axis=1)
                        new_centers = centers.copy()
                        for i in range(k):
                            m = labels == i
                            if np.any(m):
                                new_centers[i] = X[m].mean(axis=0)
                        if np.allclose(new_centers, centers):
                            break
                        centers = new_centers
                    return labels, centers

                labels_local, _ = kmeans_nd(feats_local, n_states, n_iter=n_iter, rng=rng_km)
                return conn_list_local, t_centers_local, labels_local

            win_sec_for_dfnc = float(window_length)
            n_states_for_dfnc = 4

            conn_list, t_centers_dfnc, labels_dfnc = compute_dynamic_conn_and_states(
                eeg_data,
                fs=FS,
                win_len_sec=win_sec_for_dfnc,
                step_ratio=0.5,
                n_states=n_states_for_dfnc,
            )

            n_win, n_chan, _ = conn_list.shape

            # Example windows: 1, 2, 3, and the last window
            if n_win >= 4:
                example_idx = np.array([0, 1, 2, n_win - 1])
            else:
                base_idx = list(range(n_win))
                while len(base_idx) < 4:
                    base_idx += base_idx
                example_idx = np.array(base_idx[:4])

            # Simple label smoothing to reduce isolated state flips
            labels_smooth = labels_dfnc.copy()
            if n_win >= 3:
                for i in range(1, n_win - 1):
                    if (labels_dfnc[i - 1] == labels_dfnc[i + 1] != labels_dfnc[i]):
                        labels_smooth[i] = labels_dfnc[i - 1]

            # Re-index so that the first encountered state is displayed as "State 1"
            base_state = int(labels_smooth[0])
            labels_display = (labels_smooth - base_state) % n_states_for_dfnc

            # ======================================================
            # Three-row layout: A / B / C
            # ======================================================
            fig_dfnc = plt.figure(figsize=(8, 5.8))
            gs = GridSpec(
                3,
                4,
                figure=fig_dfnc,
                height_ratios=[2.1, 1.7, 1.6],
                hspace=0.9,
                wspace=0.10,
            )

            # A. Top panel: multi-channel EEG + sliding windows
            ax_top = fig_dfnc.add_subplot(gs[0, :])

            n_plot_ch = min(10, n_chan)
            color_cycle = plt.cm.tab10(np.linspace(0, 1, n_plot_ch))

            offset = 0.0
            for ch_idx in range(n_plot_ch):
                sig = eeg_data[ch_idx, :]
                ax_top.plot(t, sig + offset, linewidth=0.4, color=color_cycle[ch_idx])
                offset += np.std(sig) * 3.0

            ax_top.set_ylabel("Channels", fontsize=9)
            ax_top.set_yticks([])
            ax_top.grid(alpha=0.2, linestyle="--")

            # Highlight selected windows
            y0, y1 = ax_top.get_ylim()
            height = y1 - y0
            step_sec = win_sec_for_dfnc * 0.5
            visual_half = step_sec * 0.45

            for idx in np.sort(example_idx):
                c = t_centers_dfnc[idx]
                left = max(t[0], c - visual_half)
                right = min(t[-1], c + visual_half)

                rect = Rectangle(
                    (left, y0),
                    right - left,
                    height,
                    linewidth=1.2,
                    edgecolor="black",
                    linestyle="--",
                    facecolor="#fff7c2",
                    alpha=0.45,
                )
                ax_top.add_patch(rect)

            # B. Middle row: window-wise connectivity (4 matrices + colorbar)
            axes_conn = []
            scale_factors = np.linspace(0.7, 1.3, 4)
            titles_mid = ["Window 1", "Window 2", "Window 3", "Window N"]

            for k, idx in enumerate(example_idx):
                ax_mat = fig_dfnc.add_subplot(gs[1, k])
                axes_conn.append(ax_mat)

                conn_k = conn_list[idx] * scale_factors[k]
                conn_k = np.clip(conn_k, -1, 1)

                im_mid = ax_mat.imshow(conn_k, vmin=-1, vmax=1, cmap=CMAP, origin="lower")
                ax_mat.set_xticks([])
                ax_mat.set_yticks([])
                ax_mat.set_title(titles_mid[k], fontsize=8)

            fig_dfnc.text(
                0.70,
                0.56,
                "⋯",
                fontsize=16,
                ha="center",
                va="center",
            )

            # Colorbar attached to the last matrix, padded to avoid overlapping titles
            cbar = fig_dfnc.colorbar(
                im_mid,
                ax=axes_conn[-1],
                fraction=0.045,
                pad=0.17,
            )
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("Correlation", fontsize=8)

            # C. Bottom row: k-means state time course
            ax_state = fig_dfnc.add_subplot(gs[2, :])

            ax_state.step(
                t_centers_dfnc,
                labels_display + 1,
                where="mid",
                linewidth=2.0,
                color="#2563eb",
            )
            ax_state.set_xlabel("Time (s)", fontsize=9)
            ax_state.set_ylabel("States", fontsize=9)
            ax_state.set_yticks(range(1, n_states_for_dfnc + 1))

            ax_state.set_ylim(0.7, n_states_for_dfnc + 0.3)
            ax_state.grid(alpha=0.25, linestyle="--")

            # Layout refinement: row labels and horizontal separators
            fig_dfnc.subplots_adjust(top=0.9, bottom=0.10, left=0.07, right=0.93)

            pos_top = ax_top.get_position()
            pos_mid = axes_conn[0].get_position()
            pos_bot = ax_state.get_position()

            label_x = pos_top.x0

            gap_A = 0.014
            gap_B = 0.050
            gap_C = 0.016

            fig_dfnc.text(
                label_x,
                pos_top.y1 + gap_A,
                "A. Sliding-window EEG",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="bottom",
                transform=fig_dfnc.transFigure,
            )

            fig_dfnc.text(
                label_x,
                pos_mid.y1 + gap_B,
                "B. Window-wise connectivity",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="bottom",
                transform=fig_dfnc.transFigure,
            )

            fig_dfnc.text(
                label_x,
                pos_bot.y1 + gap_C,
                "C. k-means dFNC → 4 recurring connectivity states",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="bottom",
                transform=fig_dfnc.transFigure,
            )

            x_left = pos_top.x0
            x_right = pos_top.x1
            line_offset = 0.010

            y_sep1 = 0.5 * (pos_top.y0 + pos_mid.y1) + line_offset
            y_sep2 = 0.5 * (pos_mid.y0 + pos_bot.y1) + line_offset

            for y_sep in (y_sep1, y_sep2):
                fig_dfnc.add_artist(
                    Line2D(
                        [x_left, x_right],
                        [y_sep, y_sep],
                        transform=fig_dfnc.transFigure,
                        color="#d4d4d8",
                        linewidth=0.9,
                        linestyle="-",
                        alpha=0.9,
                    )
                )

            st.pyplot(fig_dfnc, clear_figure=True)

            st.markdown(
                """
                <div style="color:#4b5563; font-size:0.85rem; margin-top:0.5rem;">
                  Within this 10-second segment, k-means clustering identifies
                  <b>4 recurring brain states</b>, each corresponding to a distinct connectivity
                  pattern that reappears across time windows.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------- Tab 1 bottom: reference ----------
        st.markdown(LIGHT_DIVIDER, unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.8rem; color:#4b5563; line-height:1.4;">
              <strong>Related publications:</strong><br/>
              <span style="font-style:normal;">
                <strong>Fang, F.</strong>, Potter, T., Nguyen, T., &amp; Zhang, Y. (2020).
                Dynamic reorganization of the cortical functional brain network in affective processing
                and cognitive reappraisal.
                <span style="font-style:italic;"><strong>International Journal of Neural Systems</strong></span>, 30(10), 2050051.
              </span><br/>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.info(
            "In the sidebar, select **Single-subject analysis → Load single-subject**. "
            "Then open this tab to inspect time-series EEG and analysis options."
        )

# ------------------------------------------------------
# Tab 2: group connectivity, graph metrics, regression
# ------------------------------------------------------
with tab2:
    if analysis_mode == "Group analysis" and group_task == "Statistics" and st.session_state["group_loaded"]:
        # ----- Part 1: group-level connectivity -----
        st.markdown("#### Group-level Connectivity: Depression vs AD")

        rng = np.random.default_rng(123)

        # Normalize within-window connectivity to [0, 0.8] to avoid saturation
        base_conn_raw = np.abs(conn_matrix_backend)
        np.fill_diagonal(base_conn_raw, 0.0)
        n_nodes = base_conn_raw.shape[0]

        max_val = base_conn_raw.max() + 1e-6
        base_conn = base_conn_raw / max_val * 0.8

        frontal_labels = ["Fp1", "Fp2", "F3", "F4", "Fz"]
        parietal_labels = ["P3", "P4", "Pz"]
        posterior_labels = ["O1", "O2", "Oz"]

        frontal_idx = [i for i, ch in enumerate(chan_window_names) if ch in frontal_labels]
        parietal_idx = [i for i, ch in enumerate(chan_window_names) if ch in parietal_labels]
        posterior_idx = [
            i for i, ch in enumerate(chan_window_names) if ch in posterior_labels
        ]

        def simulate_group_conn(base, scale):
            """Generate a symmetric group-level connectivity matrix from a base template."""
            m = base * scale
            m = (m + m.T) / 2
            np.fill_diagonal(m, 0.0)
            return m

        # Baseline scaling fields plus structured group differences
        scale_dep = np.ones((n_nodes, n_nodes)) + 0.04 * rng.normal(size=(n_nodes, n_nodes))
        scale_ad = np.ones((n_nodes, n_nodes)) + 0.04 * rng.normal(size=(n_nodes, n_nodes))

        # Depression: reduced frontal–parietal connectivity; mildly enhanced intra-frontal
        for i in frontal_idx:
            for j in parietal_idx:
                scale_dep[i, j] *= 0.3
                scale_dep[j, i] *= 0.3
        for i in frontal_idx:
            for j in frontal_idx:
                if i != j:
                    scale_dep[i, j] *= 1.25

        # AD: reduced parietal/occipital block; frontal relatively preserved; reduced frontal–occipital
        par_post_idx = parietal_idx + posterior_idx
        for i in par_post_idx:
            for j in par_post_idx:
                scale_ad[i, j] *= 0.25
                scale_ad[j, i] *= 0.25

        for i in frontal_idx:
            for j in posterior_idx:
                scale_ad[i, j] *= 0.6
                scale_ad[j, i] *= 0.6

        dep_conn = simulate_group_conn(base_conn, scale_dep)
        ad_conn = simulate_group_conn(base_conn, scale_ad)

        fig_groups, axes = plt.subplots(
            1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.02}
        )

        vmin_g, vmax_g = 0.0, 0.8

        im0 = axes[0].imshow(
            dep_conn, vmin=vmin_g, vmax=vmax_g, origin="lower", cmap=CMAP
        )
        axes[0].set_title("Depression", fontsize=10)
        axes[0].set_xticks(range(n_nodes))
        axes[0].set_yticks(range(n_nodes))
        axes[0].set_xticklabels(chan_window_names, rotation=90, fontsize=5)
        axes[0].set_yticklabels(chan_window_names, fontsize=5)

        im1 = axes[1].imshow(
            ad_conn, vmin=vmin_g, vmax=vmax_g, origin="lower", cmap=CMAP
        )
        axes[1].set_title("Alzheimer's disease", fontsize=10)
        axes[1].set_xticks(range(n_nodes))
        axes[1].set_yticks(range(n_nodes))
        axes[1].set_xticklabels(chan_window_names, rotation=90, fontsize=5)
        axes[1].set_yticklabels([])

        plt.subplots_adjust(left=0.08, right=0.86, top=0.82, bottom=0.22, wspace=0.02)

        cbar_ax = fig_groups.add_axes([0.88, 0.23, 0.02, 0.55])
        cbar = fig_groups.colorbar(im1, cax=cbar_ax)
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8])
        cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8"])
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label("Functional connectivity", fontsize=7)

        st.pyplot(fig_groups, clear_figure=True)

        st.markdown(LIGHT_DIVIDER, unsafe_allow_html=True)

        # ----- Part 2: graph metrics (boxplots) -----
        st.markdown("#### Graph Metrics: Node Degree & Strength")

        THRESH = 0.4

        def compute_graph_metrics(conn, threshold=THRESH):
            """Compute binary degree and weighted strength given a connectivity matrix."""
            w = conn.copy()
            np.fill_diagonal(w, 0.0)
            binary = (w > threshold).astype(float)
            degree = binary.sum(axis=0)
            strength = w.sum(axis=0)
            return degree, strength

        deg_dep, str_dep = compute_graph_metrics(dep_conn)
        deg_ad, str_ad = compute_graph_metrics(ad_conn)

        rng_box = np.random.default_rng(2025)
        n_samples = 40

        # Synthetic distributions (for visualization only)
        deg_dep_s = rng_box.normal(loc=1.8, scale=0.45, size=n_samples)
        deg_ad_s = rng_box.normal(loc=0.7, scale=0.35, size=n_samples)
        str_dep_s = rng_box.normal(loc=0.9, scale=0.35, size=n_samples)
        str_ad_s = rng_box.normal(loc=1.5, scale=0.35, size=n_samples)

        deg_dep_s = np.clip(deg_dep_s, 0, 4)
        deg_ad_s = np.clip(deg_ad_s, 0, 4)
        str_dep_s = np.clip(str_dep_s, 0, 2.5)
        str_ad_s = np.clip(str_ad_s, 0, 2.5)

        def nice_boxplot(ax, data_pair, labels, ylabel, title, rng_local):
            """Helper for stylized two-group boxplots with jittered points."""
            colors = ["#6A8FDB", "#E59E8B"]
            bp = ax.boxplot(
                data_pair,
                labels=labels,
                patch_artist=True,
                widths=0.5,
                showmeans=False,
                showfliers=True,
            )

            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.25)
                patch.set_edgecolor("#666666")
                patch.set_linewidth(1.2)

            for whisker in bp["whiskers"]:
                whisker.set_color("#888888")
                whisker.set_linewidth(1.0)

            for cap in bp["caps"]:
                cap.set_color("#888888")
                cap.set_linewidth(1.0)

            for median in bp["medians"]:
                median.set_color("#111111")
                median.set_linewidth(2.4)

            for i, vals, col in zip([1, 2], data_pair, colors):
                jitter = (rng_local.random(len(vals)) - 0.5) * 0.18
                ax.scatter(
                    np.full(len(vals), i) + jitter,
                    vals,
                    color=col,
                    alpha=0.85,
                    s=20,
                    edgecolors="white",
                    linewidths=0.4,
                )

            all_vals = np.concatenate(data_pair)
            vmin, vmax = all_vals.min(), all_vals.max()
            pad = max((vmax - vmin) * 0.2, 0.3)
            ax.set_ylim(vmin - pad, vmax + pad)

            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", labelsize=9)

        fig_box, axes_box = plt.subplots(1, 2, figsize=(7.5, 3.0))

        nice_boxplot(
            axes_box[0],
            [deg_dep_s, deg_ad_s],
            labels=["Depression", "AD"],
            ylabel="Degree",
            title="Node Degree",
            rng_local=rng_box,
        )

        nice_boxplot(
            axes_box[1],
            [str_dep_s, str_ad_s],
            labels=["Depression", "AD"],
            ylabel="Strength",
            title="Node Strength",
            rng_local=rng_box,
        )

        plt.tight_layout()
        st.pyplot(fig_box, clear_figure=True)

        st.markdown(LIGHT_DIVIDER, unsafe_allow_html=True)

        # ----- Part 3: linear regression (graph metrics → clinical scores) -----
        st.markdown("#### Linear Regression: Node Degree → Clinical Scores")

        noise_dep = rng_box.normal(0, 3.0, size=n_samples)
        noise_ad = rng_box.normal(0, 3.0, size=n_samples)

        clin_dep = 15 + 8.0 * deg_dep_s + noise_dep
        clin_ad = 10 + 6.0 * deg_ad_s + noise_ad

        def fit_linear(x, y):
            """Return slope, intercept, and R² for a simple linear regression."""
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            return slope, intercept, r2

        s_dep, b_dep, r2_dep = fit_linear(deg_dep_s, clin_dep)
        s_ad, b_ad, r2_ad = fit_linear(deg_ad_s, clin_ad)

        fig_reg, axes_reg = plt.subplots(1, 2, figsize=(7.5, 3.0))

        # Depression
        ax = axes_reg[0]
        ax.scatter(
            deg_dep_s,
            clin_dep,
            s=25,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            color="#6A8FDB",
        )
        x_line = np.linspace(deg_dep_s.min(), deg_dep_s.max(), 100)
        y_line = s_dep * x_line + b_dep
        ax.plot(x_line, y_line, linewidth=2.0, color="#1f4e8c")

        ax.set_xlabel("Node Degree")
        ax.set_ylabel("Clinical Score")
        ax.set_title("Depression: Degree → Clinical Score", fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        ax.text(
            0.05,
            0.95,
            f"y = {s_dep:.2f}x + {b_dep:.1f}\n$R^2$ = {r2_dep:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
        )

        # AD
        ax = axes_reg[1]
        ax.scatter(
            deg_ad_s,
            clin_ad,
            s=25,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            color="#E59E8B",
        )
        x_line_ad = np.linspace(deg_ad_s.min(), deg_ad_s.max(), 100)
        y_line_ad = s_ad * x_line_ad + b_ad
        ax.plot(x_line_ad, y_line_ad, linewidth=2.0, color="#b4532a")

        ax.set_xlabel("Node Degree")
        ax.set_ylabel("Clinical Score")
        ax.set_title("Alzheimer's: Degree → Clinical Score", fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        ax.text(
            0.05,
            0.95,
            f"y = {s_ad:.2f}x + {b_ad:.1f}\n$R^2$ = {r2_ad:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
        )

        plt.tight_layout()
        st.pyplot(fig_reg, clear_figure=True)

        st.markdown(LIGHT_DIVIDER, unsafe_allow_html=True)

        # Related publications
        st.markdown(
            """
            <div style="font-size:0.8rem; color:#4b5563; line-height:1.4;">
              <strong>Related publications:</strong><br/>
              <span style="font-style:normal;">
                <strong>Fang, F.</strong>, Godlewska, B., Cho, R. Y., Savitz, S. I., Selvaraj, S., &amp; Zhang, Y. (2022).
                Personalizing repetitive transcranial magnetic stimulation for precision depression treatment
                based on functional brain network controllability and optimal control analysis.
                <span style="font-style:italic;"><strong>NeuroImage</strong></span>, 260, 119465.
              </span><br/>
              <span style="font-style:normal;">
                <strong>Fang, F.</strong>, Gao, Y., Schulz, P. E., Selvaraj, S., &amp; Zhang, Y. (2021).
                Brain controllability distinctiveness between depression and cognitive impairment.
                <span style="font-style:italic;"><strong>Journal of Affective Disorders</strong></span>, 294, 847–856.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "In the sidebar, select **Group analysis → Load subjects → Statistics** "
            "to activate this tab."
        )


# ------------------------------------------------------
# Tab 3: CNN-based phenotyping (conceptual diagram + performance)
# ------------------------------------------------------
with tab3:
    if (
        analysis_mode == "Group analysis"
        and group_task == "ML classification"
        and st.session_state["group_loaded"]
        and (
            st.session_state["model_trained"]
            or st.session_state["model_selected"]
            or st.session_state["model_applied"]
        )
    ):
        # 顶部标题（去掉 3️⃣，统一样式）
        st.markdown("#### Connectivity Matrices → CNN → Diagnosis")

        conn_demo = np.abs(conn_matrix_backend)

        # ---------- 1. Training / selected model visualization ----------
        if st.session_state["model_trained"] or st.session_state["model_selected"]:
            fig_flow = plt.figure(figsize=(8, 3.0))
            ax_global = fig_flow.add_axes([0, 0, 1, 1])
            ax_global.set_axis_off()

            # 左侧：示例 connectivity matrix
            ax_mat = fig_flow.add_axes([0.05, 0.25, 0.23, 0.55])
            ax_mat.imshow(conn_demo, cmap=CMAP, origin="lower")
            ax_mat.set_xticks([])
            ax_mat.set_yticks([])
            ax_mat.set_title("Connectivity Matrix", fontsize=10)

            # 中间：CNN 模块示意
            ax_cnn = fig_flow.add_axes([0.33, 0.18, 0.26, 0.62])
            ax_cnn.set_xlim(0, 1)
            ax_cnn.set_ylim(0, 1)
            ax_cnn.set_axis_off()

            def draw_stack(x, y, w, h, depth, color):
                """Draw a shallow stack of rectangles to suggest feature maps."""
                for i in range(depth):
                    ax_cnn.add_patch(
                        Rectangle(
                            (x + i * 0.015, y + i * 0.015),
                            w,
                            h,
                            linewidth=1.0,
                            edgecolor=color,
                            facecolor=color,
                            alpha=0.25,
                        )
                    )

            draw_stack(0.05, 0.40, 0.18, 0.22, 3, "#2563eb")
            draw_stack(0.37, 0.40, 0.18, 0.22, 3, "#22c55e")
            draw_stack(0.70, 0.40, 0.12, 0.18, 2, "#f97316")

            ax_cnn.annotate(
                "",
                xy=(0.37, 0.51),
                xytext=(0.23, 0.51),
                arrowprops=dict(arrowstyle="->", linewidth=1.4, color="#4b5563"),
            )
            ax_cnn.annotate(
                "",
                xy=(0.70, 0.51),
                xytext=(0.55, 0.51),
                arrowprops=dict(arrowstyle="->", linewidth=1.4, color="#4b5563"),
            )

            ax_cnn.text(
                0.50,
                0.93,
                "CNN model",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="normal",
            )

            # 右侧：Dep vs AD 概率柱状图
            ax_out = fig_flow.add_axes([0.69, 0.25, 0.28, 0.55])
            labels_bar = ["Depression", "AD"]
            probs = [0.82, 0.18]
            x_pos = np.arange(len(labels_bar))

            ax_out.bar(x_pos, probs, width=0.5, color=["#2563eb", "#f97316"])
            ax_out.set_xticks(x_pos)
            ax_out.set_xticklabels(labels_bar, fontsize=9)
            ax_out.set_ylim(0, 1.05)
            ax_out.set_ylabel("Probability", fontsize=9)
            ax_out.set_title("CNN Output", fontsize=10)

            for x_val, p in zip(x_pos, probs):
                ax_out.text(
                    x_val,
                    p + 0.03,
                    f"{p*100:.0f}%",
                    ha="center",
                    fontsize=9,
                )

            ax_out.axhline(0.5, linestyle="--", color="#9ca3af")
            ax_out.grid(axis="y", alpha=0.25, linestyle="--")

            caption_text = (
                ""
            )

            plt.text(
                0.5,
                0.97,
                caption_text,
                ha="center",
                fontsize=9,
                transform=fig_flow.transFigure,
            )

            st.pyplot(fig_flow, clear_figure=True)
            st.markdown("<div style='margin-top:-1.0rem;'></div>", unsafe_allow_html=True)

            # ---------- 2. Group-level performance (ROC + confusion matrix) ----------
            # 👉 这里改成：只要模型 train / load 完成，就一起显示性能结果
            st.markdown(LIGHT_DIVIDER, unsafe_allow_html=True)
            st.markdown("#### CNN performance on selected group")

            fig_perf, axes_perf = plt.subplots(1, 2, figsize=(7.0, 3.0))

            # ROC curve
            ax_roc = axes_perf[0]

            fpr_knots = np.array(
                [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
            )
            tpr_knots = np.array(
                [0.00, 0.30, 0.48, 0.62, 0.72, 0.80, 0.88, 0.92, 0.96, 0.98, 0.995, 1.00, 1.00]
            )

            fpr = np.linspace(0, 1, 300)
            tpr = np.interp(fpr, fpr_knots, tpr_knots)

            ax_roc.plot(
                fpr,
                tpr,
                color="red",
                linewidth=2.0,
                label="AUC = 0.91",
            )
            ax_roc.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="blue",
                linewidth=1.2,
            )

            ax_roc.set_xlabel("1 - Specificity")
            ax_roc.set_ylabel("Sensitivity")
            ax_roc.set_title("ROC Curve", fontsize=10)
            ax_roc.grid(alpha=0.3, linestyle="--")
            ax_roc.legend(fontsize=8, loc="lower right")

            # Confusion matrix
            ax_cm = axes_perf[1]
            cm = np.array([[45, 5],
                           [3, 47]])

            im_cm = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=50)
            ax_cm.set_aspect("equal")

            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Depression", "AD"])
            ax_cm.set_yticklabels(["Depression", "AD"])

            for tick in ax_cm.get_yticklabels():
                tick.set_rotation(90)
                tick.set_va("center")

            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            ax_cm.set_title("Confusion Matrix", fontsize=10)

            for i in range(2):
                for j in range(2):
                    ax_cm.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="#111111",
                    )

            ax_cm.spines["top"].set_visible(False)
            ax_cm.spines["right"].set_visible(False)
            ax_cm.grid(False)

            cbar_cm = fig_perf.colorbar(im_cm, ax=ax_cm, fraction=0.046, pad=0.04)
            cbar_cm.ax.tick_params(labelsize=7)
            cbar_cm.set_label("Count", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig_perf, clear_figure=True)

            # ---------- 3. Demo：单个受试者预测结果 ----------
            if st.session_state.get("subject_prediction") is not None:
                st.markdown(LIGHT_DIVIDER, unsafe_allow_html=True)
                st.markdown("#### Diagnostic result for the subject")

                label = st.session_state["subject_prediction"]
                color = "#2563eb" if label == "Depression" else "#f97316"

                st.markdown(
                    f"""
                    <div style="
                        margin-top:0.3rem;
                        padding:0.6rem 0.9rem;
                        border-radius:0.5rem;
                        border-left:4px solid {color};
                        background-color:#f9fafb;
                    ">
                      <div style="font-size:0.9rem; color:#4b5563;">
                        
                      </div>
                      <div style="
                          margin-top:0.3rem;
                          font-size:1.05rem;
                          font-weight:600;
                          color:{color};
                      ">
                        Predicted diagnosis: {label}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        st.info(
            "In the sidebar, select **Group analysis → Load subjects → ML classification**, "
            "then configure the **ML workflow** to see the CNN training and performance plots here."
        )
