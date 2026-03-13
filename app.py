"""
app.py — Gradio Interface for the Tabular AutoML Framework
Run: python app.py
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import io, os, sys, time, threading, json, textwrap
from pathlib import Path

# ── make sure the automl package is importable ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from automl import AutoML

# ── Gradio version compatibility check ───────────────────────────────────────
import gradio as _gr_check
_gr_version = tuple(int(x) for x in _gr_check.__version__.split(".")[:2])
_is_gradio_4_plus = _gr_version[0] >= 4
_is_gradio_6_plus = _gr_version[0] >= 6
print(f"  Gradio version: {_gr_check.__version__}")
# gr.File type param: needed in 3.x only
_FILE_KWARGS = {} if _is_gradio_4_plus else {"type": "file"}

# ── Global state ──────────────────────────────────────────────────────────────
_state: dict = {
    "automl":    None,
    "df":        None,
    "log_lines": [],
    "running":   False,
}

# ── File path compatibility helper (Gradio 3.x / 4.x / 5.x / 6.x) ──────────
def _get_filepath(file):
    """Handle gr.File output across ALL Gradio versions including 6.x."""
    if file is None:
        return None
    # Gradio 6.x: returns plain string filepath directly
    if isinstance(file, str):
        return file
    # Gradio 6.x: sometimes returns a list (multiple files)
    if isinstance(file, list):
        return file[0] if file else None
    # Gradio 3.x early: returns dict
    if isinstance(file, dict):
        return file.get("name") or file.get("path") or file.get("tmp_path")
    # Gradio 4.x / 5.x: UploadData with .path attribute
    if hasattr(file, "path"):
        return file.path
    # Gradio 3.x late: object with .name
    if hasattr(file, "name"):
        return file.name
    # Last resort
    return str(file)



PALETTE = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "border":  "#30363d",
    "accent":  "#58a6ff",
    "green":   "#3fb950",
    "yellow":  "#d29922",
    "red":     "#f85149",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=130)
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def _styled_fig(w=12, h=6):
    fig = plt.figure(figsize=(w, h), facecolor=PALETTE["bg"])
    return fig


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    ax.spines[:].set_color(PALETTE["border"])
    if title:  ax.set_title(title,  color=PALETTE["text"],   fontsize=11, pad=10, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, color=PALETTE["muted"], fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=PALETTE["muted"], fontsize=9)
    ax.tick_params(axis="x", colors=PALETTE["muted"])
    ax.tick_params(axis="y", colors=PALETTE["muted"])
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Upload & Preview
# ─────────────────────────────────────────────────────────────────────────────

def handle_upload(file):
    if file is None:
        return (gr.update(choices=[], value=None),
                gr.update(value="<p style='color:#8b949e;'>Upload a CSV to see summary.</p>"),
                gr.update(value=None))
    try:
        print(f"  [Upload] file type: {type(file)}, value: {repr(file)[:200]}")
        filepath = _get_filepath(file)
        print(f"  [Upload] resolved filepath: {filepath}")
        if filepath is None:
            return (gr.update(choices=[], value=None),
                    gr.update(value="<p style='color:red'>Could not read file. Try uploading again.</p>"),
                    gr.update(value=None))
        df = pd.read_csv(filepath)
        _state["df"] = df
        cols = df.columns.tolist()

        # Build a rich HTML preview
        n_rows, n_cols = df.shape
        missing = df.isnull().sum().sum()
        dtypes  = df.dtypes.value_counts().to_dict()
        dtype_str = ", ".join(f"{v}× {k}" for k, v in dtypes.items())

        summary_html = f"""
        <div style="font-family:'JetBrains Mono',monospace;
                    background:#161b22;border:1px solid #30363d;
                    border-radius:10px;padding:18px;color:#e6edf3;">
          <div style="display:flex;gap:30px;margin-bottom:14px;flex-wrap:wrap;">
            <div style="text-align:center;">
              <div style="font-size:28px;font-weight:700;color:#58a6ff;">{n_rows:,}</div>
              <div style="font-size:11px;color:#8b949e;">ROWS</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:28px;font-weight:700;color:#3fb950;">{n_cols}</div>
              <div style="font-size:11px;color:#8b949e;">COLUMNS</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:28px;font-weight:700;color:#d29922;">{missing:,}</div>
              <div style="font-size:11px;color:#8b949e;">MISSING CELLS</div>
            </div>
          </div>
          <div style="font-size:11px;color:#8b949e;border-top:1px solid #30363d;padding-top:10px;">
            Dtypes: {dtype_str}
          </div>
        </div>
        """

        # First 6 rows as styled HTML table
        preview_df = df.head(6)
        tbl = preview_df.to_html(index=False, border=0, classes="preview-tbl")
        table_html = f"""
        <style>
          .preview-tbl {{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px;}}
          .preview-tbl th {{background:#21262d;color:#58a6ff;padding:8px 12px;text-align:left;border-bottom:1px solid #30363d;}}
          .preview-tbl td {{padding:6px 12px;color:#e6edf3;border-bottom:1px solid #21262d;}}
          .preview-tbl tr:hover td {{background:#1c2128;}}
        </style>
        <div style="overflow-x:auto;border-radius:8px;border:1px solid #30363d;">
          {tbl}
        </div>
        """
        combined = summary_html + "<br>" + table_html
        return (gr.update(choices=cols, value=cols[-1]),
                gr.update(value=combined),
                gr.update(value=None))
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"  [Upload Error] {err_detail}")
        return (gr.update(choices=[], value=None),
                gr.update(value=f"<div style='color:#f85149;font-family:monospace;padding:12px;background:#161b22;border:1px solid #f85149;border-radius:6px;'><b>Error loading file:</b><br>{str(e)}<br><br><small>Check HF Space logs for details.</small></div>"),
                gr.update(value=None))


def show_column_stats(col_name):
    df = _state.get("df")
    if df is None or not col_name:
        return gr.update(value=None)
    s = df[col_name]

    fig = _styled_fig(10, 3.5)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.4)

    # Distribution plot
    ax1 = fig.add_subplot(gs[0])
    _ax_style(ax1, title=f"Distribution — {col_name}")
    if pd.api.types.is_numeric_dtype(s):
        clean = s.dropna()
        ax1.hist(clean, bins=30, color=PALETTE["accent"], alpha=0.85, edgecolor="none")
        ax1.axvline(clean.mean(), color=PALETTE["yellow"], lw=1.5, linestyle="--", label=f"mean={clean.mean():.2f}")
        ax1.legend(fontsize=8, labelcolor=PALETTE["muted"], facecolor=PALETTE["surface"], edgecolor=PALETTE["border"])
    else:
        vc = s.value_counts().head(12)
        bars = ax1.barh(vc.index.astype(str), vc.values, color=PALETTE["accent"], alpha=0.85)
        ax1.invert_yaxis()

    # Stats panel
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    lines = [f"dtype:    {s.dtype}",
             f"missing:  {s.isnull().sum()} ({s.isnull().mean()*100:.1f}%)",
             f"unique:   {s.nunique()}"]
    if pd.api.types.is_numeric_dtype(s):
        lines += [f"mean:     {s.mean():.3f}",
                  f"std:      {s.std():.3f}",
                  f"min:      {s.min():.3f}",
                  f"25%:      {s.quantile(.25):.3f}",
                  f"median:   {s.median():.3f}",
                  f"75%:      {s.quantile(.75):.3f}",
                  f"max:      {s.max():.3f}"]
    txt = "\n".join(lines)
    ax2.text(0.05, 0.95, txt, transform=ax2.transAxes,
             va="top", ha="left", fontsize=9.5,
             fontfamily="monospace", color=PALETTE["text"],
             bbox=dict(boxstyle="round,pad=0.6", facecolor=PALETTE["surface"],
                       edgecolor=PALETTE["border"]))

    fig.patch.set_facecolor(PALETTE["bg"])
    return gr.update(value=_fig_to_pil(fig))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Training
# ─────────────────────────────────────────────────────────────────────────────

class LogCapture:
    """Redirect stdout to both terminal and our log buffer."""
    def __init__(self, original):
        self.original = original
        self.lines = _state["log_lines"]

    def write(self, msg):
        self.original.write(msg)
        if msg.strip():
            ts = time.strftime("%H:%M:%S")
            self.lines.append(f"[{ts}] {msg.rstrip()}")

    def flush(self):
        self.original.flush()


def run_training(target_col, task_type, time_budget, n_trials,
                 use_fe, val_size, test_size, seed):
    df = _state.get("df")
    if df is None:
        return ("❌ Please upload a dataset first.", "", None, None)
    if not target_col:
        return ("❌ Please select a target column.", "", None, None)

    _state["log_lines"].clear()
    _state["running"] = True

    original_stdout = sys.stdout
    sys.stdout = LogCapture(original_stdout)

    status = "✅ Training complete!"
    try:
        budget = float(time_budget) if time_budget and float(time_budget) > 0 else None
        automl = AutoML(
            task_type=task_type,
            time_budget=budget,
            n_optuna_trials=int(n_trials),
            val_size=float(val_size),
            test_size=float(test_size),
            seed=int(seed),
            use_feature_engineering=use_fe,
            output_dir="./automl_output",
        )
        automl.fit(df, target_col=target_col)
        _state["automl"] = automl
    except Exception as e:
        status = f"❌ Error: {e}"
        import traceback; traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        _state["running"] = False

    log_text = "\n".join(_state["log_lines"])
    lb_df    = _build_leaderboard_df()
    lb_plot  = _build_leaderboard_plot()
    return (status, log_text, lb_df, lb_plot)


def _build_leaderboard_df():
    am = _state.get("automl")
    if am is None:
        return None
    df = am.leaderboard.to_dataframe()
    df = df.drop(columns=["_type", "primary_score"], errors="ignore")
    # Round floats
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(4)
    return df


def _build_leaderboard_plot():
    am = _state.get("automl")
    if am is None:
        return None

    lb = am.leaderboard.to_dataframe()
    if lb.empty:
        return None

    lb = lb.drop(columns=["_type", "primary_score"], errors="ignore")
    metric_cols = [c for c in lb.columns if c != "model_name"]
    if not metric_cols:
        return None

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics,
                             figsize=(max(4, 3.5 * n_metrics), 4.5),
                             facecolor=PALETTE["bg"])
    if n_metrics == 1:
        axes = [axes]

    colors = [PALETTE["accent"], PALETTE["green"], PALETTE["yellow"],
              "#a371f7", "#f78166", "#79c0ff", "#56d364"]

    for i, (ax, metric) in enumerate(zip(axes, metric_cols)):
        _ax_style(ax, title=metric.upper())
        vals   = lb[metric].values
        names  = lb["model_name"].values
        col    = colors[i % len(colors)]
        bars   = ax.barh(names, vals, color=col, alpha=0.85, height=0.55)
        ax.invert_yaxis()
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8, color=PALETTE["text"])
        ax.set_xlim(0, max(vals) * 1.18)
        # Highlight best bar
        best_idx = np.argmax(vals) if metric not in ("rmse","mae") else np.argmin(vals)
        axes[i].get_children()[best_idx].set_color(PALETTE["green"])

    fig.suptitle("Model Leaderboard — All Metrics", color=PALETTE["text"],
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_pil(fig)


def poll_log():
    """Stream log lines while training is running."""
    return "\n".join(_state["log_lines"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Results & Metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_results_tab():
    am = _state.get("automl")
    if am is None:
        return (gr.update(value="No training run yet."),
                gr.update(value=None),
                gr.update(value=None))

    # Summary card
    best_name    = am.best_model_name
    best_metrics = am.best_metrics
    task         = am.task_type

    metrics_rows = "".join(
        f"""<div style="display:flex;justify-content:space-between;
                        border-bottom:1px solid #30363d;padding:6px 0;">
              <span style="color:#8b949e;">{k}</span>
              <span style="color:#58a6ff;font-weight:600;">{v:.4f}</span>
            </div>"""
        for k, v in best_metrics.items()
    )
    card = f"""
    <div style="font-family:'JetBrains Mono',monospace;
                background:#161b22;border:1px solid #3fb950;
                border-radius:10px;padding:20px;color:#e6edf3;max-width:480px;">
      <div style="font-size:11px;color:#3fb950;letter-spacing:2px;margin-bottom:6px;">
        🏆 BEST MODEL
      </div>
      <div style="font-size:22px;font-weight:700;margin-bottom:16px;">{best_name}</div>
      <div style="font-size:11px;color:#8b949e;margin-bottom:8px;">TASK: {task.upper()}</div>
      {metrics_rows}
    </div>
    """

    # Metric radar / bar chart
    fig = _build_metrics_radar(best_metrics, task)
    radar_img = _fig_to_pil(fig)

    # Learning curves for PyTorch models
    curves_img = _build_loss_curves()

    return (gr.update(value=card),
            gr.update(value=radar_img),
            gr.update(value=curves_img))


def _build_metrics_radar(metrics, task):
    names = list(metrics.keys())
    vals  = list(metrics.values())

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=PALETTE["bg"])
    _ax_style(ax, title=f"Best Model Metrics")

    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=[PALETTE["accent"], PALETTE["green"],
                                   PALETTE["yellow"], "#a371f7", "#f78166"][:len(names)],
                  alpha=0.85, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.2)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8.5,
                color=PALETTE["text"])

    fig.tight_layout()
    return fig


def _build_loss_curves():
    am = _state.get("automl")
    if am is None:
        return None

    # Collect trainers from leaderboard
    trainers = [(e["model_name"], e["_model"])
                for e in am.leaderboard.entries
                if e.get("_type") == "pytorch" and hasattr(e["_model"], "history")]
    if not trainers:
        return None

    fig, axes = plt.subplots(1, len(trainers),
                             figsize=(5.5 * len(trainers), 4),
                             facecolor=PALETTE["bg"])
    if len(trainers) == 1:
        axes = [axes]

    for ax, (name, trainer) in zip(axes, trainers):
        _ax_style(ax, title=f"{name} — Loss Curves",
                  xlabel="Epoch", ylabel="Loss")
        h = trainer.history
        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], color=PALETTE["accent"],
                lw=1.8, label="Train")
        ax.plot(epochs, h["val_loss"], color=PALETTE["green"],
                lw=1.8, linestyle="--", label="Validation")
        ax.legend(fontsize=8, labelcolor=PALETTE["muted"],
                  facecolor=PALETTE["surface"], edgecolor=PALETTE["border"])
        ax.fill_between(epochs, h["train_loss"], alpha=0.08, color=PALETTE["accent"])
        ax.fill_between(epochs, h["val_loss"],   alpha=0.08, color=PALETTE["green"])

    fig.tight_layout()
    return _fig_to_pil(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def build_importance_tab(top_k):
    am = _state.get("automl")
    if am is None or not am.feature_importance:
        return gr.update(value=None)

    importance = am.feature_importance
    items = list(importance.items())[:int(top_k)]
    names = [i[0] for i in items]
    vals  = [i[1] for i in items]

    fig = _styled_fig(10, max(4, len(names) * 0.45))
    ax  = fig.add_subplot(111)
    _ax_style(ax, title=f"Top {len(names)} Feature Importances", xlabel="Importance Score")

    colors_grad = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))[::-1]
    bars = ax.barh(names[::-1], vals[::-1], color=colors_grad, alpha=0.9, height=0.65)

    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8, color=PALETTE["text"])

    ax.set_xlim(0, max(vals) * 1.18)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    fig.tight_layout()
    return gr.update(value=_fig_to_pil(fig))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Predict on New Data
# ─────────────────────────────────────────────────────────────────────────────

def predict_on_file(file):
    am = _state.get("automl")
    if am is None:
        return (None, "<p style='color:red'>No trained model. Run training first.</p>")
    if file is None:
        return (None, "<p style='color:red'>Upload a CSV to predict on.</p>")
    try:
        filepath = _get_filepath(file)
        new_df = pd.read_csv(filepath)
        preds  = am.predict(new_df)
        new_df["prediction"] = preds
        out_path = "./automl_output/predictions.csv"
        os.makedirs("./automl_output", exist_ok=True)
        new_df.to_csv(out_path, index=False)

        n = len(preds)
        if am.task_type == "regression":
            html = f"""
            <div style="font-family:monospace;background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:16px;color:#e6edf3;">
              <b>✅ {n} predictions generated</b><br><br>
              Mean: {preds.mean():.3f} &nbsp;|&nbsp;
              Std: {preds.std():.3f} &nbsp;|&nbsp;
              Min: {preds.min():.3f} &nbsp;|&nbsp;
              Max: {preds.max():.3f}
            </div>"""
        else:
            unique, counts = np.unique(preds, return_counts=True)
            dist = " &nbsp;|&nbsp; ".join(f"Class {u}: {c}" for u, c in zip(unique, counts))
            html = f"""
            <div style="font-family:monospace;background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:16px;color:#e6edf3;">
              <b>✅ {n} predictions generated</b><br><br>{dist}
            </div>"""
        return (out_path, html)
    except Exception as e:
        return (None, f"<p style='color:red'>Error: {e}</p>")


def predict_manual(vals_json):
    am = _state.get("automl")
    df = _state.get("df")
    if am is None or df is None:
        return "<p style='color:red'>Train a model first.</p>"
    try:
        row = json.loads(vals_json)
        input_df = pd.DataFrame([row])
        pred = am.predict(input_df)
        val  = pred[0] if hasattr(pred, "__len__") else pred
        return f"""
        <div style="font-family:monospace;background:#161b22;border:2px solid #3fb950;
                    border-radius:8px;padding:20px;color:#e6edf3;text-align:center;">
          <div style="font-size:13px;color:#8b949e;margin-bottom:8px;">PREDICTION</div>
          <div style="font-size:36px;font-weight:700;color:#3fb950;">{val:.4f if isinstance(val, float) else val}</div>
          <div style="font-size:11px;color:#8b949e;margin-top:8px;">Model: {am.best_model_name}</div>
        </div>"""
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>"


def build_manual_input_template():
    df = _state.get("df")
    am = _state.get("automl")
    if df is None or am is None:
        return "{}"
    feature_cols = [c for c in df.columns if c != am.best_model_name]
    sample = df.drop(columns=[am.best_model_name] if am.best_model_name in df.columns else [],
                     errors="ignore").iloc[0].to_dict()
    # Clean non-serialisable types
    clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
             for k, v in sample.items()
             if not isinstance(v, float) or not np.isnan(v)}
    return json.dumps(clean, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Dataset Analysis Visuals
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_plots():
    df = _state.get("df")
    if df is None:
        return (None, None, None)

    # 1. Missing values heatmap style bar
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    fig1 = _styled_fig(10, max(3, len(missing) * 0.4 + 1.5))
    if not missing.empty:
        ax = fig1.add_subplot(111)
        _ax_style(ax, title="Missing Values by Column (%)", xlabel="Missing %")
        cols_m  = [PALETTE["red"] if v > 20 else PALETTE["yellow"] if v > 5 else PALETTE["accent"]
                   for v in missing.values]
        bars = ax.barh(missing.index, missing.values, color=cols_m, alpha=0.85)
        for bar, v in zip(bars, missing.values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{v:.1f}%", va="center", fontsize=8, color=PALETTE["text"])
        ax.set_xlim(0, max(missing.values) * 1.25)
        red_p   = mpatches.Patch(color=PALETTE["red"],    label=">20%")
        yel_p   = mpatches.Patch(color=PALETTE["yellow"], label="5-20%")
        blue_p  = mpatches.Patch(color=PALETTE["accent"], label="<5%")
        ax.legend(handles=[red_p, yel_p, blue_p], fontsize=8,
                  facecolor=PALETTE["surface"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["muted"])
    else:
        ax = fig1.add_subplot(111)
        _ax_style(ax, title="Missing Values")
        ax.text(0.5, 0.5, "✅  No missing values!", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=PALETTE["green"])
    fig1.tight_layout()

    # 2. Correlation matrix (numeric only)
    num_df = df.select_dtypes("number")
    fig2   = _styled_fig(9, 7)
    ax2    = fig2.add_subplot(111)
    if len(num_df.columns) >= 2:
        corr = num_df.corr()
        im   = ax2.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)
        ticks = range(len(corr.columns))
        ax2.set_xticks(ticks); ax2.set_yticks(ticks)
        ax2.set_xticklabels(corr.columns, rotation=45, ha="right",
                            fontsize=7, color=PALETTE["muted"])
        ax2.set_yticklabels(corr.columns, fontsize=7, color=PALETTE["muted"])
        ax2.set_title("Correlation Matrix", color=PALETTE["text"], fontsize=11,
                      fontweight="bold", pad=10)
        ax2.set_facecolor(PALETTE["surface"])
    else:
        _ax_style(ax2, title="Correlation Matrix")
        ax2.text(0.5, 0.5, "Need ≥2 numeric columns", transform=ax2.transAxes,
                 ha="center", va="center", color=PALETTE["muted"])
    fig2.patch.set_facecolor(PALETTE["bg"])
    fig2.tight_layout()

    # 3. Data types pie
    dtypes_count = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            dtypes_count["Numeric"] = dtypes_count.get("Numeric", 0) + 1
        elif pd.api.types.is_object_dtype(df[col]):
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > 30:
                dtypes_count["Text"] = dtypes_count.get("Text", 0) + 1
            else:
                dtypes_count["Categorical"] = dtypes_count.get("Categorical", 0) + 1
        else:
            dtypes_count["Other"] = dtypes_count.get("Other", 0) + 1

    fig3 = _styled_fig(5.5, 4.5)
    ax3  = fig3.add_subplot(111)
    ax3.set_facecolor(PALETTE["bg"])
    wedge_colors = [PALETTE["accent"], PALETTE["green"],
                    PALETTE["yellow"], PALETTE["red"]][:len(dtypes_count)]
    wedges, texts, autotexts = ax3.pie(
        dtypes_count.values(),
        labels=dtypes_count.keys(),
        colors=wedge_colors,
        autopct="%1.0f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=2),
    )
    for t in texts:      t.set_color(PALETTE["muted"]); t.set_fontsize(10)
    for t in autotexts:  t.set_color(PALETTE["bg"]);    t.set_fontsize(9)
    ax3.set_title("Feature Type Distribution", color=PALETTE["text"],
                  fontsize=11, fontweight="bold", pad=10)
    fig3.patch.set_facecolor(PALETTE["bg"])

    return (_fig_to_pil(fig1), _fig_to_pil(fig2), _fig_to_pil(fig3))


# ─────────────────────────────────────────────────────────────────────────────
# Build the Gradio App
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ───────────────────────────────── */
body, .gradio-container {
  background: #0d1117 !important;
  font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
  color: #e6edf3 !important;
}

/* ── Header ───────────────────────────────── */
.app-header {
  background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
  border-bottom: 1px solid #30363d;
  padding: 28px 32px 20px;
  margin-bottom: 8px;
}
.app-title {
  font-size: 28px;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(90deg, #58a6ff, #3fb950);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.app-subtitle {
  color: #8b949e;
  font-size: 12px;
  margin-top: 4px;
  letter-spacing: 1px;
}

/* ── Tabs ─────────────────────────────────── */
.tab-nav button {
  background: transparent !important;
  color: #8b949e !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  font-family: inherit !important;
  font-size: 12px !important;
  padding: 10px 18px !important;
  letter-spacing: 0.5px;
}
.tab-nav button.selected {
  color: #58a6ff !important;
  border-bottom-color: #58a6ff !important;
}

/* ── Inputs & Textboxes ───────────────────── */
input, textarea, select, .gr-input, .gr-textarea {
  background: #161b22 !important;
  border: 1px solid #30363d !important;
  color: #e6edf3 !important;
  border-radius: 6px !important;
  font-family: inherit !important;
}
input:focus, textarea:focus {
  border-color: #58a6ff !important;
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

/* ── Buttons ──────────────────────────────── */
.gr-button-primary, button.primary {
  background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
  color: white !important;
  border: none !important;
  border-radius: 6px !important;
  font-family: inherit !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 10px 22px !important;
  letter-spacing: 0.3px;
  transition: opacity 0.2s !important;
}
.gr-button-primary:hover { opacity: 0.88 !important; }

.gr-button-secondary, button.secondary {
  background: #21262d !important;
  color: #e6edf3 !important;
  border: 1px solid #30363d !important;
  border-radius: 6px !important;
  font-family: inherit !important;
}

/* ── Dropdown ─────────────────────────────── */
.gr-dropdown select { background: #161b22 !important; color: #e6edf3 !important; }

/* ── Slider ───────────────────────────────── */
.gr-slider input[type=range] { accent-color: #58a6ff; }

/* ── Blocks / Panels ──────────────────────── */
.gr-block, .gr-panel, .gr-box {
  background: #161b22 !important;
  border: 1px solid #30363d !important;
  border-radius: 8px !important;
}

/* ── Log textbox ──────────────────────────── */
.log-box textarea {
  background: #0d1117 !important;
  color: #3fb950 !important;
  font-size: 11px !important;
  font-family: 'JetBrains Mono', monospace !important;
  border: 1px solid #30363d !important;
}

/* ── Status badge ─────────────────────────── */
.status-box textarea {
  background: #161b22 !important;
  color: #58a6ff !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  border: 1px solid #30363d !important;
  text-align: center !important;
}

/* ── DataFrame ────────────────────────────── */
.gr-dataframe table {
  background: #161b22 !important;
  color: #e6edf3 !important;
  font-size: 12px !important;
  font-family: inherit !important;
}
.gr-dataframe th {
  background: #21262d !important;
  color: #58a6ff !important;
  border-bottom: 1px solid #30363d !important;
}
.gr-dataframe td { border-bottom: 1px solid #21262d !important; }

/* ── Labels ───────────────────────────────── */
label, .gr-form label, .gr-label {
  color: #8b949e !important;
  font-size: 11px !important;
  letter-spacing: 0.5px !important;
  text-transform: uppercase !important;
}

/* ── Accordion ────────────────────────────── */
.gr-accordion { border: 1px solid #30363d !important; border-radius: 8px !important; }
"""

HEADER_HTML = """
<div class="app-header">
  <div class="app-title">⚡ Tabular AutoML</div>
  <div class="app-subtitle">AUTOMATED MACHINE LEARNING · CLASSICAL + NEURAL MODELS · BAYESIAN HPO</div>
</div>
"""


def build_app():
    # Blocks title param available since gradio 3.9
    try:
        blocks_kwargs = dict(css=CUSTOM_CSS, title="AutoML Studio")
        gr.Blocks(**blocks_kwargs)  # test
    except TypeError:
        blocks_kwargs = dict(css=CUSTOM_CSS)

    with gr.Blocks(**blocks_kwargs) as app:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            # TAB 1 — Upload & Explore
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("📂  Upload & Explore"):
                gr.Markdown("### Upload your CSV dataset to get started")

                with gr.Row():
                    with gr.Column(scale=1):
                        upload_btn = gr.File(label="📂 Drop CSV here (or click to browse)")
                        target_dd  = gr.Dropdown(label="Target Column", choices=[], interactive=True)
                        explore_btn = gr.Button("🔍 Analyze Selected Column", variant="secondary")

                    with gr.Column(scale=2):
                        dataset_summary = gr.HTML(value="<p style='color:#8b949e;'>Upload a CSV to see summary.</p>")

                col_plot = gr.Image(label="Column Distribution")

                upload_btn.change(
                    fn=handle_upload,
                    inputs=[upload_btn],
                    outputs=[target_dd, dataset_summary, col_plot]
                )
                explore_btn.click(
                    fn=show_column_stats,
                    inputs=[target_dd],
                    outputs=[col_plot]
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 2 — Configure & Train
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🚀  Configure & Train"):
                gr.Markdown("### Training Configuration")

                with gr.Row():
                    with gr.Column(scale=1):
                        task_radio   = gr.Radio(["classification", "regression"],
                                                label="Task Type", value="regression")
                        time_budget  = gr.Number(label="Time Budget (seconds, blank = unlimited)",
                                                 value=300, precision=0)
                        n_trials     = gr.Slider(3, 30, value=15, step=1,
                                                 label="Optuna HPO Trials per Model")
                        use_fe       = gr.Checkbox(label="Enable Feature Engineering", value=True)

                    with gr.Column(scale=1):
                        val_size     = gr.Slider(0.05, 0.3, value=0.15, step=0.01,
                                                 label="Validation Split Size")
                        test_size    = gr.Slider(0.05, 0.3, value=0.15, step=0.01,
                                                 label="Test Split Size")
                        seed_in      = gr.Number(label="Random Seed", value=42, precision=0)
                        train_btn    = gr.Button("▶  Start Training", variant="primary")

                with gr.Row():
                    train_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("📋 Training Log", open=True):
                    log_box = gr.Textbox(label="Live Output", lines=18,
                                         interactive=False)
                    refresh_btn = gr.Button("↻ Refresh Log", variant="secondary")

                gr.Markdown("### Leaderboard Preview")
                lb_table = gr.DataFrame(label="Model Scores")
                lb_plot  = gr.Image(label="Metric Comparison")

                train_btn.click(
                    fn=run_training,
                    inputs=[target_dd, task_radio, time_budget, n_trials,
                            use_fe, val_size, test_size, seed_in],
                    outputs=[train_status, log_box, lb_table, lb_plot]
                )
                refresh_btn.click(fn=poll_log, inputs=[], outputs=[log_box])

            # ══════════════════════════════════════════════════════════════
            # TAB 3 — Results & Metrics
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("📊  Results & Metrics"):
                gr.Markdown("### Best Model Performance")
                results_btn = gr.Button("Load Results", variant="secondary")

                with gr.Row():
                    with gr.Column(scale=1):
                        best_card  = gr.HTML(value="<p style='color:#8b949e;'>Run training first.</p>")
                    with gr.Column(scale=1):
                        metrics_plot = gr.Image(label="Metrics Chart")

                gr.Markdown("### PyTorch Training Curves")
                curves_plot = gr.Image(label="Loss Curves (neural models only)")

                results_btn.click(
                    fn=build_results_tab,
                    inputs=[],
                    outputs=[best_card, metrics_plot, curves_plot]
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 4 — Feature Importance
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🔍  Feature Importance"):
                gr.Markdown("### SHAP / Model-Based Feature Importance")
                with gr.Row():
                    top_k_slider = gr.Slider(5, 40, value=15, step=1, label="Top K Features")
                    imp_btn      = gr.Button("Generate Importance Plot", variant="primary")

                importance_plot = gr.Image(label="Feature Importances")

                imp_btn.click(
                    fn=build_importance_tab,
                    inputs=[top_k_slider],
                    outputs=[importance_plot]
                )
                top_k_slider.change(
                    fn=build_importance_tab,
                    inputs=[top_k_slider],
                    outputs=[importance_plot]
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 5 — Predict
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🎯  Predict"):
                gr.Markdown("### Batch Prediction (CSV file)")
                with gr.Row():
                    with gr.Column():
                        pred_file   = gr.File(label="Upload CSV for prediction")
                        pred_btn    = gr.Button("Run Prediction", variant="primary")
                    with gr.Column():
                        pred_result = gr.HTML()
                        pred_dl     = gr.File(label="Download Predictions CSV")

                pred_btn.click(
                    fn=predict_on_file,
                    inputs=[pred_file],
                    outputs=[pred_dl, pred_result]
                )

                gr.Markdown("---")
                gr.Markdown("### Manual Single-Row Prediction")
                gr.Markdown("Paste a JSON object with feature values:")
                with gr.Row():
                    with gr.Column():
                        template_btn  = gr.Button("📋 Load Sample Row Template", variant="secondary")
                        manual_json   = gr.Textbox(label="Input JSON", lines=10,
                                                   placeholder='{"feature_0": 1.23, "cat_A": "high", ...}')
                        manual_btn    = gr.Button("Predict", variant="primary")
                    with gr.Column():
                        manual_out    = gr.HTML()

                template_btn.click(
                    fn=build_manual_input_template,
                    inputs=[], outputs=[manual_json]
                )
                manual_btn.click(
                    fn=predict_manual,
                    inputs=[manual_json], outputs=[manual_out]
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 6 — Dataset Analysis
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🧬  Dataset Analysis"):
                gr.Markdown("### Automated Dataset Visualizations")
                analysis_btn = gr.Button("Generate Analysis Plots", variant="primary")

                with gr.Row():
                    missing_plot = gr.Image(label="Missing Values")

                with gr.Row():
                    corr_plot    = gr.Image(label="Correlation Matrix")
                    dtype_plot   = gr.Image(label="Feature Types")

                analysis_btn.click(
                    fn=build_analysis_plots,
                    inputs=[],
                    outputs=[missing_plot, corr_plot, dtype_plot]
                )

        # Footer
        gr.HTML("""
        <div style="text-align:center;padding:20px;
                    border-top:1px solid #30363d;
                    color:#8b949e;font-size:11px;
                    letter-spacing:0.5px;margin-top:16px;">
          TABULAR AUTOML · SKLEARN + PYTORCH · OPTUNA HPO · SHAP EXPLAINABILITY
        </div>
        """)

    return app


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AutoML Gradio Interface")
    print(f"  Gradio version: {_gr_check.__version__}")
    print("="*55 + "\n")
    app = build_app()

    # Detect if running on HF Spaces
    import os
    on_hf_spaces = os.environ.get("SPACE_ID") is not None

    if on_hf_spaces:
        # HF Spaces: minimal launch args
        app.launch(ssr_mode=False)
    else:
        # Local: full args
        launch_kwargs = dict(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
        )
        # ssr_mode only exists in Gradio 5+
        if _gr_version[0] >= 5:
            launch_kwargs["ssr_mode"] = False
        app.launch(**launch_kwargs)
