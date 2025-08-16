"""
Unified reporting utilities to standardize outputs across Logistic Regression, Random Forest,
XGBoost, and CatBoost for the OULAD student-failure prediction project.

What this file does
-------------------
- 数据集描述：样本量、特征数、positive 占比（CSV）
- 模型性能对比（5 折交叉验证）：F1_positive、Precision、Recall、AUC-PR（均值±标准差）（CSV + 可选图）
- PR 曲线：每个模型的 Precision-Recall 曲线（典型一次分割，或全量 holdout）（PNG）
- 阈值优化结果：阈值=0.3, 0.4, 0.5 的 Precision、Recall、F1（CSV + 折线图PNG）
- 特征重要性：RF/XGB/CB 的 top 10（CSV + 条形图PNG）；Logistic 可输出 top 系数（可选）
- 解释性分析（可选）：SHAP summary（PNG）或 PDP（PNG）
- 错误分析：混淆矩阵（PNG）、FP/FN 占比（CSV）、典型样本统计（CSV）
- 业务场景模拟（可选）：不同 recall 下预计干预人数（CSV）

How to use
----------
1) 放在你的项目中，例如 src/reporting/unified_reporting.py
2) 在训练脚本中导入并调用 `run_full_report(...)` 或用更细的函数自己拼装。
3) 需要的三方库：numpy, pandas, scikit-learn, matplotlib, (可选) shap, xgboost, catboost

Notes
-----
- 本脚本仅负责评估与产物落盘，不改变你的训练流程。
- 你可以传入自定义的 CV splitter（例如 GroupKFold 基于学生ID），确保不泄漏同一学生。
"""
from __future__ import annotations
import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import matplotlib.pyplot as plt

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# === Utility dataclasses ===
@dataclass
class CVResult:
    model_name: str
    f1_pos_mean: float
    f1_pos_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    aucpr_mean: float
    aucpr_std: float


# === Filesystem helpers ===
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def _save_fig(path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# === 1) Dataset summary ===
def export_dataset_summary(X: pd.DataFrame, y: pd.Series, out_csv: str) -> pd.DataFrame:
    """导出数据集描述：样本量、特征数、positive 占比"""
    n_samples = len(y)
    n_features = X.shape[1]
    pos_ratio = float(np.mean(y))
    df = pd.DataFrame({
        "n_samples": [n_samples],
        "n_features": [n_features],
        "positive_ratio": [pos_ratio],
    })
    _save_csv(df, out_csv)
    return df


# === 2) Cross-validated metrics (mean ± std) ===

def cross_val_metrics(
    model_dict: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: Optional[Iterable[Tuple[np.ndarray, np.ndarray]]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    artifacts_dir: str = "artifacts",
    csv_name: str = "cv_metrics.csv",
) -> pd.DataFrame:
    """对每个模型做 Stratified K-Fold（或自定义cv_splits），输出 F1_pos/Precision/Recall/AUC-PR 的均值与方差。
    注意：正类记为 y==1。
    """
    if cv_splits is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splits = skf.split(X, y)

    rows: List[CVResult] = []

    # 为可复用，先将 splits 物化
    splits = [(train_idx, test_idx) for train_idx, test_idx in cv_splits]

    for name, base_model in model_dict.items():
        f1s, precs, recs, aucprs = [], [], [], []

        for train_idx, valid_idx in splits:
            X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

            model = clone(base_model)
            model.fit(X_tr, y_tr)

            # proba / decision function
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_va)[:, 1]
            elif hasattr(model, "decision_function"):
                p = model.decision_function(X_va)
                # 将 margin 缩放到 [0,1] 以便可比（sigmoid）
                p = 1 / (1 + np.exp(-p))
            else:
                # 兜底：用预测标签当作概率（不推荐）
                p = model.predict(X_va)

            # 默认阈值 0.5 计算 P/R/F1
            y_pred = (p >= 0.5).astype(int)
            precs.append(precision_score(y_va, y_pred, zero_division=0))
            recs.append(recall_score(y_va, y_pred, zero_division=0))
            f1s.append(f1_score(y_va, y_pred, zero_division=0))
            aucprs.append(average_precision_score(y_va, p))

        rows.append(
            CVResult(
                name,
                float(np.mean(f1s)), float(np.std(f1s)),
                float(np.mean(precs)), float(np.std(precs)),
                float(np.mean(recs)), float(np.std(recs)),
                float(np.mean(aucprs)), float(np.std(aucprs)),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    _save_csv(df, os.path.join(artifacts_dir, csv_name))
    return df


# === 3) Precision-Recall curve export ===

def export_pr_curve(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    out_png: str,
    title: Optional[str] = None,
) -> None:
    model_name = getattr(model, "__class__", type(model)).__name__
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        p = model.decision_function(X_test)
        p = 1 / (1 + np.exp(-p))
    else:
        p = model.predict(X_test)

    precision, recall, _ = precision_recall_curve(y_test, p)
    ap = average_precision_score(y_test, p)

    plt.figure(figsize=(5, 4))
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    ttl = title or f"PR Curve — {model_name} (AP={ap:.3f})"
    plt.title(ttl)
    _save_fig(out_png)


# === 4) Threshold sweep ===

def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: List[float],
    artifacts_dir: str,
    csv_name: str,
    out_png: Optional[str] = None,
    model_label: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    df = pd.DataFrame(rows)
    _save_csv(df, os.path.join(artifacts_dir, csv_name))

    if out_png is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(df["threshold"], df["precision"], marker="o", label="Precision")
        plt.plot(df["threshold"], df["recall"], marker="o", label="Recall")
        plt.plot(df["threshold"], df["f1"], marker="o", label="F1")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        ttl = f"Threshold sweep — {model_label or ''}".strip()
        plt.title(ttl)
        plt.legend()
        _save_fig(out_png)

    return df


# === 5) Feature importance ===

def export_feature_importance(
    model,
    feature_names: List[str],
    top_k: int,
    artifacts_dir: str,
    csv_name: str,
    out_png: Optional[str] = None,
    title: Optional[str] = None,
) -> pd.DataFrame:
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.squeeze(np.asarray(model.coef_))
        imp = np.abs(coef)
    elif hasattr(model, "get_feature_importance"):
       imp = np.asarray(model.get_feature_importance(), dtype=float)
    else:
        warnings.warn("Model has neither feature_importances_ nor coef_. Skipping.")
        return pd.DataFrame()

    top_k = min(top_k, len(feature_names))
    idx = np.argsort(imp)[::-1][:top_k]
    df = pd.DataFrame({
        "feature": [feature_names[i] for i in idx],
        "importance": imp[idx],
    })
    _save_csv(df, os.path.join(artifacts_dir, csv_name))

    if out_png is not None and len(df) > 0:
        plt.figure(figsize=(6, 4))
        plt.barh(range(len(df)), df["importance"][::-1])
        plt.yticks(range(len(df)), df["feature"][::-1])
        plt.xlabel("Importance")
        plt.title(title or "Top feature importance")
        _save_fig(out_png)

    return df


# === 6) Optional SHAP summary or PDP ===

def shap_summary_plot(model, X_sample: pd.DataFrame, out_png: str, title: Optional[str] = None) -> None:
    if not _HAS_SHAP:
        warnings.warn("shap not installed; skipping SHAP plot")
        return
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        plt.figure(figsize=(6, 4))
        shap.summary_plot(shap_values, X_sample, show=False)
        if title:
            plt.title(title)
        _save_fig(out_png)
    except Exception as e:
        warnings.warn(f"SHAP failed: {e}")


def pdp_plot(model, X: pd.DataFrame, feature: str, out_png: str, grid_resolution: int = 20) -> None:
    try:
        from sklearn.inspection import partial_dependence
        vals = partial_dependence(model, X, [feature], grid_resolution=grid_resolution)
        xs = vals["values"][0]
        ys = vals["average"][0]
        plt.figure(figsize=(5, 4))
        plt.plot(xs, ys)
        plt.xlabel(feature)
        plt.ylabel("Partial dependence")
        plt.title(f"PDP — {feature}")
        _save_fig(out_png)
    except Exception as e:
        warnings.warn(f"PDP failed: {e}")


# === 7) Error analysis ===

def error_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    artifacts_dir: str,
    cm_png: str,
    rates_csv: str,
    exemplar_csv: Optional[str] = None,
    X_test: Optional[pd.DataFrame] = None,
    top_k: int = 50,
) -> Tuple[np.ndarray, pd.DataFrame]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 保存混淆矩阵图
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    _save_fig(os.path.join(artifacts_dir, cm_png))

    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    rates = pd.DataFrame([
        {"metric": "FP_rate", "value": fp / max(total, 1)},
        {"metric": "FN_rate", "value": fn / max(total, 1)},
        {"metric": "TP_rate", "value": tp / max(total, 1)},
        {"metric": "TN_rate", "value": tn / max(total, 1)},
    ])
    _save_csv(rates, os.path.join(artifacts_dir, rates_csv))

    # 典型样本（最高分的FN/最低分的FP）
    if exemplar_csv is not None and X_test is not None:
        df = X_test.copy()
        df["y_true"] = y_true
        df["y_proba"] = y_proba
        df["y_pred"] = y_pred
        fn_df = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].sort_values("y_proba", ascending=False).head(top_k)
        fp_df = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].sort_values("y_proba", ascending=True).head(top_k)
        fn_df["type"], fp_df["type"] = "FN", "FP"
        ex = pd.concat([fn_df, fp_df], axis=0)
        _save_csv(ex, os.path.join(artifacts_dir, exemplar_csv))

    return cm, rates


# === 8) Business scenario simulation ===

def simulate_intervention(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    recall_targets: List[float],
    artifacts_dir: str,
    out_csv: str,
) -> pd.DataFrame:
    """给定若干 recall 目标，找到达到该 recall 的最小阈值，并估算干预人数（预测为1的人数）。"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall 是按阈值降序的曲线。为简便，逐点搜索。
    rows = []
    for r in recall_targets:
        # 找到 recall >= r 的最小阈值索引
        idx = np.where(recall >= r)[0]
        if len(idx) == 0:
            rows.append({"target_recall": r, "achievable": False})
            continue
        i = idx[-1]  # recall 数组从高到低变化，取最后一个满足条件的点
        prec = float(precision[i])
        rec = float(recall[i])
        # 估算预测为正的人数：P(ŷ=1) ≈ (TP+FP) = TP/prec; 其中 TP = rec * (#positive)
        n_pos = int(np.sum(y_true == 1))
        tp_est = rec * n_pos
        pred_pos_est = tp_est / max(prec, 1e-9)
        rows.append({
            "target_recall": r,
            "achievable": True,
            "precision": prec,
            "recall": rec,
            "estimated_predicted_positive": int(round(pred_pos_est)),
        })
    df = pd.DataFrame(rows)
    _save_csv(df, os.path.join(artifacts_dir, out_csv))
    return df


# === Orchestrator ===

def run_full_report(
    model_dict: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None,
    artifacts_dir: str = "artifacts",
    cv_splits: Optional[Iterable[Tuple[np.ndarray, np.ndarray]]] = None,
    thresholds: List[float] = [0.3, 0.4, 0.5],
    recall_targets: List[float] = [0.5, 0.6, 0.7, 0.8],
    do_shap: bool = False,
    shap_sample: int = 5000,
    pdp_features: Optional[List[str]] = None,
) -> None:
    _ensure_dir(artifacts_dir)

    # 1) Dataset summary
    export_dataset_summary(X_train, y_train, os.path.join(artifacts_dir, "dataset_summary.csv"))

    # 2) CV metrics
    cv_df = cross_val_metrics(
        model_dict, pd.concat([X_train, X_test], axis=0), pd.concat([y_train, y_test], axis=0),
        cv_splits=cv_splits, artifacts_dir=artifacts_dir
    )

    # 逐模型 fit 全量训练，导出 PR/阈值/重要性/错误分析
    feature_names = feature_names or list(X_train.columns)

    for name, model in model_dict.items():
        model.fit(X_train, y_train)

        # 3) PR curve
        export_pr_curve(model, X_test, y_test, os.path.join(artifacts_dir, f"pr_curve_{name}.png"), title=f"PR — {name}")

        # Get probabilities on test
        if hasattr(model, "predict_proba"):
            p_test = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            p_test = model.decision_function(X_test)
            p_test = 1 / (1 + np.exp(-p_test))
        else:
            p_test = model.predict(X_test)

        # 4) Threshold sweep
        threshold_sweep(
            y_true=y_test.values,
            y_proba=p_test,
            thresholds=thresholds,
            artifacts_dir=artifacts_dir,
            csv_name=f"thresholds_{name}.csv",
            out_png=os.path.join(artifacts_dir, f"thresholds_{name}.png"),
            model_label=name,
        )

        # 5) Feature importance
        export_feature_importance(
            model,
            feature_names=feature_names,
            top_k=10,
            artifacts_dir=artifacts_dir,
            csv_name=f"feature_importance_top10_{name}.csv",
            out_png=os.path.join(artifacts_dir, f"feature_importance_top10_{name}.png"),
            title=f"Top 10 features — {name}",
        )

        # 6) SHAP summary or PDP (optional)
        if do_shap:
            if X_train.shape[0] > shap_sample:
                X_sample = X_train.sample(shap_sample, random_state=42)
            else:
                X_sample = X_train
            shap_summary_plot(model, X_sample, os.path.join(artifacts_dir, f"shap_summary_{name}.png"), title=f"SHAP — {name}")

        if pdp_features:
            for f in pdp_features[:2]:  # 限制 1-2 个
                pdp_plot(model, X_test, f, os.path.join(artifacts_dir, f"pdp_{name}_{f}.png"))

        # 7) Error analysis at 0.5
        error_analysis(
            y_true=y_test.values,
            y_proba=p_test,
            threshold=0.5,
            artifacts_dir=artifacts_dir,
            cm_png=f"confusion_matrix_{name}.png",
            rates_csv=f"error_rates_{name}.csv",
            exemplar_csv=f"error_exemplars_{name}.csv",
            X_test=X_test,
            top_k=50,
        )

        # 8) Business scenario simulation
        simulate_intervention(
            y_true=y_test.values,
            y_proba=p_test,
            recall_targets=recall_targets,
            artifacts_dir=artifacts_dir,
            out_csv=f"intervention_{name}.csv",
        )

    # 保存一次性的汇总元数据
    meta = {
        "models": list(model_dict.keys()),
        "thresholds": thresholds,
        "recall_targets": recall_targets,
    }
    with open(os.path.join(artifacts_dir, "report_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# === Example (paste into your training script) ===
if __name__ == "__main__":
    # 这是一个最小使用示例：请在你的训练脚本里替换为真实数据与模型
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    try:
        from xgboost import XGBClassifier  # type: ignore
        _HAS_XGB = True
    except Exception:
        _HAS_XGB = False

    try:
        from catboost import CatBoostClassifier  # type: ignore
        _HAS_CB = True
    except Exception:
        _HAS_CB = False

    X, y = make_classification(n_samples=5000, n_features=60, weights=[0.85, 0.15], random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y)

    # 简单 holdout
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Logistic": LogisticRegression(max_iter=200, n_jobs=None),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    if _HAS_XGB:
        models["XGB"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)
    if _HAS_CB:
        models["CB"] = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, loss_function="Logloss", verbose=False, random_state=42)

    run_full_report(
        model_dict=models,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        artifacts_dir="artifacts",
        do_shap=False,
        pdp_features=None,
    )
