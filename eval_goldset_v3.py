#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval script v3 (single-table input)

Input:
  - one xlsx that already contains:
    - sample_id
    - model_pred_json
    - GT columns:
        order_outcome_gt_value, order_outcome_gt_state
        paid_amount_gt_value, paid_amount_gt_state
        original_amount_gt_value, original_amount_gt_state
        discount_amount_gt_value, discount_amount_gt_state
        order_time_gt_value, order_time_gt_state
        reasoning_gt_result
    - bucket columns (for bucket metrics):
        domain, page_state, info_completeness

Output:
  - report_3.0.xlsx with:
      summary
      error_cases
      bucket_by_domain
      bucket_by_pagestate
      bucket_by_infocompleteness
      debug_uniques (unique values for bucket columns)
"""

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def _safe_float(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if s == "":
        return None
    s2 = re.sub(r"[^0-9\.\-]", "", s)
    if s2 in ("", ".", "-", "-."):
        return None
    try:
        return float(s2)
    except Exception:
        return None


def parse_pred_json(x):
    """
    Parse model_pred_json cell into dict.
    Handles:
    - already a dict
    - a json string
    - an escaped json string like "{""a"":1}"
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, dict):
        return x
    s = str(x).strip()
    if not s:
        return None
    # common Excel escaped JSON
    if s.startswith('"{') and s.endswith('}"'):
        s = s[1:-1]
        s = s.replace('""', '"')
    try:
        return json.loads(s)
    except Exception:
        return None


FIELDS = ["order_outcome", "paid_amount", "original_amount", "discount_amount", "order_time"]


def field_is_correct(gt_value, gt_state, pred_obj):
    """
    Return True/False/None:
    - None means cannot judge (pred missing or schema broken)
    Strict:
      - gt_state must match pred.state
      - if gt_state==OK: compare value string-normalized
      - else: pred.value must be null-ish
    """
    if pred_obj is None or not isinstance(pred_obj, dict):
        return None

    ps = pred_obj.get("state", None)
    pv = pred_obj.get("value", None)

    if gt_state not in ("OK", "MISSING", "UNCLEAR"):
        return None
    if ps not in ("OK", "MISSING", "UNCLEAR"):
        return False

    if gt_state != ps:
        return False

    if gt_state == "OK":
        if gt_value is None or (isinstance(gt_value, float) and math.isnan(gt_value)):
            return False
        g = str(gt_value).strip()
        p = "" if pv is None else str(pv).strip()

        # numeric fields: compare float within tolerance
        g_num = re.sub(r"[^\d\.\-]", "", g)
        p_num = re.sub(r"[^\d\.\-]", "", p)
        if re.fullmatch(r"-?\d+(\.\d+)?", g_num) and re.fullmatch(r"-?\d+(\.\d+)?", p_num):
            gf = _safe_float(g)
            pf = _safe_float(p)
            if gf is None or pf is None:
                return False
            return abs(gf - pf) < 1e-6

        # otherwise string exact match (e.g., time)
        return g == p

    else:
        return pv is None or (isinstance(pv, float) and math.isnan(pv)) or str(pv).strip().lower() in ("null", "")


def reasoning_is_correct(gt_result, pred_reasoning):
    if pred_reasoning is None or not isinstance(pred_reasoning, dict):
        return None
    pr = pred_reasoning.get("result", None)
    if gt_result not in ("YES", "NO", "SKIP"):
        return None
    return pr == gt_result


def schema_ok(pred):
    """
    Check top-level schema minimal requirements.
    """
    if pred is None or not isinstance(pred, dict):
        return False
    for k in FIELDS + ["reasoning"]:
        if k not in pred:
            return False
    for k in FIELDS:
        v = pred.get(k)
        if not isinstance(v, dict):
            return False
        if "state" not in v or "value" not in v or "evidence" not in v:
            return False
        if v.get("state") not in ("OK", "MISSING", "UNCLEAR"):
            return False
    r = pred.get("reasoning")
    if not isinstance(r, dict):
        return False
    if "result" not in r or "explain" not in r:
        return False
    if r.get("result") not in ("YES", "NO", "SKIP"):
        return False
    return True


def compute_metrics(df_eval: pd.DataFrame) -> pd.DataFrame:
    n = len(df_eval)
    if n == 0:
        return pd.DataFrame([{
            "n": 0,
            "schema_ok_rate": None,
            "all_5_fields_acc": None,
            "reasoning_acc": None,
            "order_outcome_acc": None,
            "paid_amount_acc": None,
            "original_amount_acc": None,
            "discount_amount_acc": None,
            "order_time_acc": None,
        }])

    def rate(col):
        s = df_eval[col].dropna()
        if len(s) == 0:
            return None
        return float(s.mean())

    row = {
        "n": n,
        "schema_ok_rate": rate("schema_ok"),
        "all_5_fields_acc": rate("all5_ok"),
        "reasoning_acc": rate("reasoning_ok"),
    }
    for f in FIELDS:
        row[f"{f}_acc"] = rate(f"{f}_ok")

    out = pd.DataFrame([row])
    # keep column names consistent with your report
    out = out.rename(columns={
        "order_outcome_acc": "order_outcome_acc",
        "paid_amount_acc": "paid_amount_acc",
        "original_amount_acc": "original_amount_acc",
        "discount_amount_acc": "discount_amount_acc",
        "order_time_acc": "order_time_acc",
    })
    return out


def make_bucket_table(df_eval: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df_eval.columns:
        return pd.DataFrame()

    rows = []
    for g, sdf in df_eval.groupby(group_col, dropna=False):
        m = compute_metrics(sdf).iloc[0].to_dict()
        m[group_col] = g
        rows.append(m)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    cols = [group_col] + [c for c in out.columns if c != group_col]
    return out[cols].sort_values(by=[group_col]).reset_index(drop=True)


def require_columns(df: pd.DataFrame, cols, where="input table"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {where}: {missing}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Single xlsx containing GT columns + model_pred_json.")
    ap.add_argument("--out", default="report_3.0.xlsx", help="Output report xlsx.")
    args = ap.parse_args()

    df = pd.read_excel(args.pred)

    # required columns
    gt_required = []
    for f in FIELDS:
        gt_required += [f"{f}_gt_value", f"{f}_gt_state"]
    base_required = ["sample_id", "model_pred_json", "reasoning_gt_result", "domain", "page_state", "info_completeness"]

    require_columns(df, base_required + gt_required, where=args.pred)

    # parse predictions
    df["_pred"] = df["model_pred_json"].apply(parse_pred_json)
    df["schema_ok"] = df["_pred"].apply(schema_ok)

    # per-field correctness
    for f in FIELDS:
        gv = df[f"{f}_gt_value"]
        gs = df[f"{f}_gt_state"]
        df[f"{f}_ok"] = [
            field_is_correct(gv.iloc[i], gs.iloc[i], (df["_pred"].iloc[i] or {}).get(f))
            if df["_pred"].iloc[i] is not None else None
            for i in range(len(df))
        ]

    # all 5 fields correct
    def _all5(row):
        vals = [row.get(f"{f}_ok") for f in FIELDS]
        if any(v is None for v in vals):
            return None
        return all(bool(v) for v in vals)

    df["all5_ok"] = df.apply(_all5, axis=1)

    # reasoning
    df["reasoning_ok"] = [
        reasoning_is_correct(df["reasoning_gt_result"].iloc[i], (df["_pred"].iloc[i] or {}).get("reasoning"))
        if df["_pred"].iloc[i] is not None else None
        for i in range(len(df))
    ]

    # error flag
    def _is_error(row):
        if row["schema_ok"] is False:
            return True
        for col in ["all5_ok", "reasoning_ok"] + [f"{f}_ok" for f in FIELDS]:
            v = row.get(col)
            if v is False:
                return True
        return False

    df["is_error"] = df.apply(_is_error, axis=1)

    # summary + buckets
    summary = compute_metrics(df)
    bucket_domain = make_bucket_table(df, "domain")
    bucket_pagestate = make_bucket_table(df, "page_state")
    bucket_infocomp = make_bucket_table(df, "info_completeness")

    # debug uniques (to prevent “why is it still old tags?”)
    debug_uniques = pd.DataFrame({
        "domain_uniques": pd.Series(sorted(df["domain"].dropna().astype(str).unique())),
        "page_state_uniques": pd.Series(sorted(df["page_state"].dropna().astype(str).unique())),
        "info_completeness_uniques": pd.Series(sorted(df["info_completeness"].dropna().astype(str).unique())),
    })

    # write report
    out_path = Path(args.out)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="summary")

        err_cols = [c for c in [
            "sample_id","domain","page_state","info_completeness",
            "schema_ok","all5_ok","reasoning_ok",
            "order_outcome_ok","paid_amount_ok","original_amount_ok","discount_amount_ok","order_time_ok",
            "model_pred_json"
        ] if c in df.columns]
        df[df["is_error"]].to_excel(w, index=False, sheet_name="error_cases", columns=err_cols)

        bucket_domain.to_excel(w, index=False, sheet_name="bucket_by_domain")
        bucket_pagestate.to_excel(w, index=False, sheet_name="bucket_by_pagestate")
        bucket_infocomp.to_excel(w, index=False, sheet_name="bucket_by_infocompleteness")

        debug_uniques.to_excel(w, index=False, sheet_name="debug_uniques")

    print(f"✅ Done. Saved: {out_path.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()