"""
IEEE 14-Bus ML Load Flow Analyzer — Backend
============================================
Memory-friendly approach: only the small preprocessing scalers stay in RAM
across requests.  ML models are loaded one-at-a-time per request, used to
predict over the entire input batch, then immediately freed before the next
model is loaded.  This caps peak RAM at "scalers + 1 model" instead of
"scalers + 7 models", which is what allowed the 7-model dashboard to launch
on machines that previously OOM'd with all-models-resident.

Pandapower (NR solver) still runs in an isolated subprocess because it is
not thread-safe and can corrupt shared state.
"""

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
import pandas as pd
import numpy as np
import joblib
import io
import os
import gc
import multiprocessing as mp
from contextlib import asynccontextmanager
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ML_ASSETS: dict[str, Any] = {}

VALID_TOPOLOGY_VALUES = set(range(32))
MAX_BATCH_ROWS   = 8760
MAX_COMPARE_ROWS = 10    # NR solver is slow; hard-cap compare-file at 10 rows
BUS1_VOLTAGE     = 1.06  # IEEE 14-bus slack bus is 1.06 pu

# ── Contingency maps ─────────────────────────────────────────────────────────
CODE_TO_LOAD: dict[int, int] = {
    20: 1, 23: 2, 24: 3, 25: 4, 26: 5,
    27: 6, 28: 7, 29: 8, 30: 9, 21: 10, 22: 11,
}
CODE_TO_LINE: dict[int, int] = {
    5: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9,
    6: 10, 7: 11, 8: 12, 9: 13, 10: 14, 11: 15,
}

# ── Model / scaler file layout ────────────────────────────────────────────────
# Models are NOT preloaded — they are streamed in one-at-a-time per request.
# Each entry: (output_key, model_filename, target-scaler key in ML_ASSETS).
MODEL_DEFS = [
    ("V_pred", "Voltage_Model.joblib",    "ts_V"),
    ("A_pred", "Angle_Model.joblib",      "ts_A"),
    ("P_send", "SendingP_Model.joblib",   "ts_P_send"),
    ("P_rec",  "ReceivingP_Model.joblib", "ts_P_rec"),
    ("I_line", "Iline_Model.joblib",      "ts_I"),
    ("Q_send", "SendingQ_Model.joblib",   "ts_Q_send"),
    ("Q_rec",  "ReceivingQ_Model.joblib", "ts_Q_rec"),
]

SCALER_FILES = {
    "num_scaler":  "numerical_scaler.pkl",
    "cat_encoder": "categorical_encoder.pkl",
    "ts_V":        "V2_V14_target_scaler.pkl",
    "ts_A":        "Angle_2_to_14_target_scaler.pkl",
    "ts_P_send":   "SendingPs_target_scaler.pkl",
    "ts_P_rec":    "ReceivingPs_target_scaler.pkl",
    "ts_I":        "Line_Currents_target_scaler.pkl",
    "ts_Q_send":   "SendingQs_target_scaler.pkl",
    "ts_Q_rec":    "ReceivingQs_target_scaler.pkl",
}


# ═══════════════════════════════════════════════════════════════
#  STARTUP / SHUTDOWN  — load everything once
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """At startup we only load the small scalers/encoders into RAM. The heavy
    ML models are loaded on demand per request (see _load_predict_inverse)."""
    S_DIR = os.path.join(BASE_DIR, "scalers")
    M_DIR = os.path.join(BASE_DIR, "models")

    print("Loading scalers...")
    for key, fname in SCALER_FILES.items():
        ML_ASSETS[key] = joblib.load(os.path.join(S_DIR, fname))
        print(f"  ✓ {fname}")

    # Sanity-check that every required model file is present on disk so we
    # fail fast at startup instead of mid-request.
    missing = [
        fname for _, fname, _ in MODEL_DEFS
        if not os.path.isfile(os.path.join(M_DIR, fname))
    ]
    if missing:
        raise RuntimeError(f"Missing model files in {M_DIR}: {missing}")

    print(f"✅ Scalers loaded ({len(ML_ASSETS)} assets). "
          f"{len(MODEL_DEFS)} models will stream-load per request. Server ready.")
    yield
    ML_ASSETS.clear()


app = FastAPI(title="Data-Driven Load Flow Analyzer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory="../frontend")
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# ═══════════════════════════════════════════════════════════════
#  INFERENCE  (fully in-process, no subprocess spawn)
# ═══════════════════════════════════════════════════════════════
def _preprocess(X_num: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
    X_ns = ML_ASSETS["num_scaler"].transform(X_num)
    X_ce = ML_ASSETS["cat_encoder"].transform(X_cat)
    return np.hstack([X_ns, X_ce]).astype(np.float32)


def _load_predict_inverse(model_filename: str, scaler_key: str,
                          X: np.ndarray, models_dir: str) -> np.ndarray:
    """Load a model, predict on the full batch, inverse-scale, then free.

    Peak extra RAM during this call is one model + one prediction array.
    The model is dropped before this function returns, so the caller never
    holds two models simultaneously.
    """
    model = joblib.load(os.path.join(models_dir, model_filename))
    try:
        raw = model.predict(X)
    finally:
        del model
        gc.collect()

    # Guarantee 2-D so sklearn scalers' inverse_transform doesn't choke.
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    elif raw.ndim > 2:
        raw = raw.reshape(raw.shape[0], -1)
    out = ML_ASSETS[scaler_key].inverse_transform(raw)
    del raw
    gc.collect()
    return out


def apply_contingency_zeroing(
    topo: int,
    ps: np.ndarray, pr: np.ndarray, il: np.ndarray,
    qs: np.ndarray, qr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero out the tripped line's flows/currents for line contingencies."""
    if topo in CODE_TO_LINE:
        idx = CODE_TO_LINE[topo] - 1  # 0-based
        ps[idx] = 0.0
        pr[idx] = 0.0
        il[idx] = 0.0
        qs[idx] = 0.0
        qr[idx] = 0.0
    return ps, pr, il, qs, qr


def run_inference(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    topology_categories: list[int],
) -> list[dict[str, Any]]:

    X = _preprocess(X_num, X_cat)
    del X_num, X_cat
    gc.collect()

    M_DIR = os.path.join(BASE_DIR, "models")

    # Stream-load every model in turn; only the current one lives in RAM.
    preds: dict[str, np.ndarray] = {}
    for out_key, fname, scaler_key in MODEL_DEFS:
        print(f"  → {fname}")
        preds[out_key] = _load_predict_inverse(fname, scaler_key, X, M_DIR)

    # X is no longer needed once every model has predicted.
    del X
    gc.collect()

    V_pred = preds["V_pred"]
    A_pred = preds["A_pred"]
    P_send = preds["P_send"]
    P_rec  = preds["P_rec"]
    I_line = preds["I_line"]
    Q_send = preds["Q_send"]
    Q_rec  = preds["Q_rec"]

    V_LOW, V_HIGH = 0.95, 1.05
    results: list[dict[str, Any]] = []

    for i, topo in enumerate(topology_categories):
        ps, pr, il, qs, qr = apply_contingency_zeroing(
            topo,
            P_send[i].copy(),
            P_rec[i].copy(),
            I_line[i].copy(),
            Q_send[i].copy(),
            Q_rec[i].copy(),
        )
        # pandapower convention: *_to is negative at the receiving end, so the
        # true line loss is sending + receiving (signed sum), for both P and Q.
        p_loss = ps + pr
        q_loss = qs + qr

        v = V_pred[i].tolist()
        voltage_violations = [
            {"bus": f"V{j+2}", "value": round(val, 4),
             "type": "low" if val < V_LOW else "high"}
            for j, val in enumerate(v)
            if val < V_LOW or val > V_HIGH
        ]

        results.append({
            "hour":              i + 1,
            "topology_category": topo,
            "V2_V14":            v,
            "Angle_2_14":        A_pred[i].tolist(),
            "P_send":            ps.tolist(),
            "P_rec":             pr.tolist(),
            "P_loss":            p_loss.tolist(),
            "Total_P_Loss":      float(np.sum(p_loss)),
            "Q_send":            qs.tolist(),
            "Q_rec":             qr.tolist(),
            "Q_loss":            q_loss.tolist(),
            "Total_Q_Loss":      float(np.sum(q_loss)),
            "Line_Currents":     il.tolist(),
            "Line_Loading_Pct":  [round(abs(float(c)) * 100, 2) for c in il],
            "voltage_violations": voltage_violations,
            "has_violations":    len(voltage_violations) > 0,
        })

    print(f"✅ Inference complete ({len(results)} rows).")
    return results


# ═══════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════
class SingleInput(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: list[float]) -> list[float]:
        if len(v) != 23:
            raise ValueError(f"Expected 23 features, got {len(v)}.")
        if int(v[22]) not in VALID_TOPOLOGY_VALUES:
            raise ValueError(f"Invalid topology: {int(v[22])}.")
        if any(np.isnan(f) for f in v[:22]):
            raise ValueError("NaN detected in feature vector.")
        return v


class CompareInput(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: list[float]) -> list[float]:
        if len(v) != 23:
            raise ValueError(f"Expected 23 features, got {len(v)}.")
        if int(v[22]) not in VALID_TOPOLOGY_VALUES:
            raise ValueError(f"Invalid topology: {int(v[22])}.")
        return v


# ═══════════════════════════════════════════════════════════════
#  PANDAPOWER WORKER  (isolated subprocess — not thread-safe)
# ═══════════════════════════════════════════════════════════════
def _run_pandapower_worker(P_vals: list, Q_vals: list, topo: int) -> dict:
    """
    Runs pandapower Newton-Raphson in a fresh subprocess.
    Returns a dict mirroring the ML result structure (V, angle, P, Q, current).
    """
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case14()

    _CTL = {20:1,23:2,24:3,25:4,26:5,27:6,28:7,29:8,30:9,21:10,22:11}
    _CTLi= {5:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8,19:9,6:10,7:11,8:12,9:13,10:14,11:15}

    if   topo in _CTLi: net.line.at[_CTLi[topo]-1, "in_service"] = False
    elif topo in _CTL:  net.load.at[_CTL[topo]-1,  "in_service"] = False
    elif 1 <= topo <= 4: net.gen.at[topo-1, "in_service"] = False
    elif topo == 31:     net.shunt.at[0,    "in_service"] = False

    net.load["p_mw"]   = P_vals
    net.load["q_mvar"] = Q_vals

    try:
        pp.runpp(net, algorithm="nr", numba=False)
    except Exception as e:
        return {"error": str(e)}

    vm = net.res_bus["vm_pu"].values.tolist()
    va = net.res_bus["va_degree"].values.tolist()
    il = net.res_line["i_ka"].values.tolist()
    pf = net.res_line["p_from_mw"].values.tolist()
    pt = net.res_line["p_to_mw"].values.tolist()
    pl = net.res_line["pl_mw"].values.tolist()
    qf = net.res_line["q_from_mvar"].values.tolist()
    qt = net.res_line["q_to_mvar"].values.tolist()
    ql = net.res_line["ql_mvar"].values.tolist()

    if topo in _CTLi:
        idx = _CTLi[topo] - 1
        for lst in (il, pf, pt, pl, qf, qt, ql):
            lst[idx] = 0.0

    return {
        "V_all":        vm,
        "A_all":        va,
        "V2_V14":       vm[1:],
        "Angle_2_14":   va[1:],
        "P_send":       pf,
        "P_rec":        pt,
        "P_loss":       pl,
        "Total_P_Loss": float(sum(pl)),
        "Q_send":       qf,
        "Q_rec":        qt,
        "Q_loss":       ql,
        "Total_Q_Loss": float(sum(ql)),
        "Line_Currents":    il,
        "Line_Loading_Pct": [round(float(c) * 100, 2) for c in il],
        "iterations":   int(net._ppc.get("iterations", -1))
                        if hasattr(net, "_ppc") and net._ppc else -1,
        "converged":    True,
    }


# ═══════════════════════════════════════════════════════════════
#  ERROR METRICS
# ═══════════════════════════════════════════════════════════════
def _compute_errors(ml: dict, nr: dict) -> dict:
    def mae(a, b):  return float(np.mean(np.abs(np.array(a) - np.array(b))))
    def rmse(a, b): return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))
    def maxe(a, b): return float(np.max(np.abs(np.array(a) - np.array(b))))

    ml_v = [BUS1_VOLTAGE] + ml["V2_V14"]
    nr_v = nr["V_all"]

    return {
        "voltage": {
            "mae":     mae(ml_v, nr_v),
            "rmse":    rmse(ml_v, nr_v),
            "max_err": maxe(ml_v, nr_v),
            "per_bus": [round(abs(ml_v[i] - nr_v[i]), 6) for i in range(14)],
        },
        "p_loss": {
            "mae":     mae(ml["P_loss"], nr["P_loss"]),
            "rmse":    rmse(ml["P_loss"], nr["P_loss"]),
            "max_err": maxe(ml["P_loss"], nr["P_loss"]),
        },
        "q_loss": {
            "mae":     mae(ml["Q_loss"], nr["Q_loss"]),
            "rmse":    rmse(ml["Q_loss"], nr["Q_loss"]),
            "max_err": maxe(ml["Q_loss"], nr["Q_loss"]),
        },
        "total_p_loss_err": round(abs(ml["Total_P_Loss"] - nr["Total_P_Loss"]), 4),
        "total_q_loss_err": round(abs(ml["Total_Q_Loss"] - nr["Total_Q_Loss"]), 4),
    }


# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html"
    )

@app.get("/health")
async def health_check():
    n = len(ML_ASSETS)
    return {"status": "ok", "assets_loaded": n, "ready": n > 0}


@app.post("/predict_single")
async def predict_single(data: SingleInput):
    X_num = np.array([data.features[:22]], dtype=np.float32)
    X_cat = np.array([[data.features[22]]])
    results = run_inference(X_num, X_cat, [int(data.features[22])])
    return {"status": "success", "total_hours": 1, "data": results}


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    allowed  = (".csv", ".xls", ".xlsx", ".parquet")
    if not any(filename.endswith(e) for e in allowed):
        raise HTTPException(400, f"Unsupported format. Allowed: {allowed}")

    contents = await file.read()
    try:
        if   filename.endswith(".csv"):              df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xls", ".xlsx")):   df = pd.read_excel(io.BytesIO(contents))
        else:                                        df = pd.read_parquet(io.BytesIO(contents))

        if df.shape[1] < 23:
            raise HTTPException(422, f"Expected ≥23 columns, got {df.shape[1]}.")
        if len(df) > MAX_BATCH_ROWS:
            raise HTTPException(422, f"Too many rows: {len(df)} (max {MAX_BATCH_ROWS}).")
        if len(df) == 0:
            raise HTTPException(422, "File contains no data.")

        df = df.iloc[:, :23]
        topo_list = df.iloc[:, 22].astype(int).tolist()
        invalid = [t for t in topo_list if t not in VALID_TOPOLOGY_VALUES]
        if invalid:
            raise HTTPException(422, f"Invalid topology values: {set(invalid)}.")

        X_num = df.iloc[:, :22].values.astype(np.float32)
        X_cat = df.iloc[:, 22].values.reshape(-1, 1)
        if np.any(np.isnan(X_num)):
            raise HTTPException(422, "NaN values in numerical columns.")

        del df, contents
        gc.collect()

        results = run_inference(X_num, X_cat, topo_list)
        return {"status": "success", "total_hours": len(results), "data": results}

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/compare")
async def compare_loadflow(data: CompareInput):
    features = data.features
    P_vals   = features[:11]
    Q_vals   = features[11:22]
    topo     = int(features[22])

    # ML inference (always succeeds)
    X_num = np.array([features[:22]], dtype=np.float32)
    X_cat = np.array([[features[22]]])
    ml = run_inference(X_num, X_cat, [topo])[0]

    # NR solver (isolated subprocess — may diverge)
    nr_converged = False
    nr = None
    nr_error = None
    try:
        with mp.get_context("spawn").Pool(1) as pool:
            nr_raw = pool.apply(_run_pandapower_worker, (P_vals, Q_vals, topo))
        if "error" in nr_raw:
            nr_error = nr_raw["error"]
        else:
            nr = nr_raw
            nr_converged = True
    except Exception as e:
        nr_error = str(e)

    if nr_converged:
        return {
            "status":       "success",
            "nr_converged": True,
            "topology":     topo,
            "ml":           ml,
            "nr":           nr,
            "errors":       _compute_errors(ml, nr),
        }
    else:
        # NR diverged — return ML-only result with divergence details
        return {
            "status":       "nr_diverged",
            "nr_converged": False,
            "topology":     topo,
            "ml":           ml,
            "nr":           None,
            "nr_error":     nr_error,
            "errors":       None,
        }


@app.post("/compare_batch")
async def compare_batch(file: UploadFile = File(...)):
    """
    Upload CSV/XLSX/Parquet for batch ML vs NR comparison.
    Only the first MAX_COMPARE_ROWS rows are processed (NR is slow).
    """
    filename = (file.filename or "").lower()
    allowed  = (".csv", ".xls", ".xlsx", ".parquet")
    if not any(filename.endswith(e) for e in allowed):
        raise HTTPException(400, f"Unsupported format. Allowed: {allowed}")

    contents = await file.read()
    try:
        if   filename.endswith(".csv"):            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xls", ".xlsx")): df = pd.read_excel(io.BytesIO(contents))
        else:                                      df = pd.read_parquet(io.BytesIO(contents))

        total_rows = len(df)
        if df.shape[1] < 23:
            raise HTTPException(422, f"Expected ≥23 columns, got {df.shape[1]}.")
        if total_rows == 0:
            raise HTTPException(422, "File contains no data.")

        df = df.iloc[:MAX_COMPARE_ROWS, :23]
        rows_used = len(df)
        topo_list = df.iloc[:, 22].astype(int).tolist()
        invalid = [t for t in topo_list if t not in VALID_TOPOLOGY_VALUES]
        if invalid:
            raise HTTPException(422, f"Invalid topology values: {set(invalid)}.")

        X_num = df.iloc[:, :22].values.astype(np.float32)
        X_cat = df.iloc[:, 22].values.reshape(-1, 1)
        ml_list = run_inference(X_num, X_cat, topo_list)

        nr_list = []
        for idx in range(rows_used):
            try:
                with mp.get_context("spawn").Pool(1) as pool:
                    nr_raw = pool.apply(_run_pandapower_worker, (
                        df.iloc[idx, :11].tolist(),
                        df.iloc[idx, 11:22].tolist(),
                        topo_list[idx],
                    ))
                nr_list.append(nr_raw)
            except Exception as e:
                nr_list.append({"error": str(e)})

        data_out = []
        for ml, nr in zip(ml_list, nr_list):
            if "error" in nr:
                # NR diverged — still include ML result
                data_out.append({
                    "topology":     ml["topology_category"],
                    "nr_converged": False,
                    "nr_error":     nr["error"],
                    "ml":           ml,
                    "nr":           None,
                    "errors":       None,
                })
            else:
                data_out.append({
                    "topology":     ml["topology_category"],
                    "nr_converged": True,
                    "ml":           ml,
                    "nr":           nr,
                    "errors":       _compute_errors(ml, nr),
                })

        return {
            "status":              "success",
            "rows_used":           rows_used,
            "total_rows_in_file":  total_rows,
            "data":                data_out,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
    