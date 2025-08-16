# src/trainers/train_rf.py
"""
Train a Random Forest pipeline on the weekly data, evaluate and save the trained model.
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src.data.load_data import load_weekly
from src.features.preprocessor import get_feature_pipeline


def main(
    data_dir: str = 'data/raw/anonymisedData',
    artifact_path: str = 'artifacts/rf_pipeline.pkl',
    test_size: float = 0.2,
    random_state: int = 42
):
    # 1. 加载并准备数据
    df = load_weekly(data_dir)
    y = df['target_fail']
    drop_cols = [
        'id_student', 'code_module', 'code_presentation',
        'week_end', 'target_fail'
    ]
    X = df.drop(columns=drop_cols)

    # 2. 定义特征类别
    num_feats = [
        *[c for c in X.columns if c.startswith('cum_click_') or c.startswith('sum_click_')],
        'week', 'week_start_day', 'num_of_prev_attempts', 'studied_credits'
    ]
    ord_feats = [c for c in X.columns if c in ('highest_education', 'imd_band', 'age_band')]
    nom_feats = [c for c in X.columns if c in ('gender', 'region', 'disability')]

    # 3. 构建预处理 + 模型 Pipeline
    preprocessor = get_feature_pipeline(
        num_feats=num_feats,
        ord_feats=ord_feats,
        nom_feats=nom_feats,
        model_type='tree'
    )
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('smote',   SMOTE(random_state=random_state)),  # imblearn 的采样步骤
        ('clf', RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ))
    ])

    # 4. 按学生 ID 分组划分训练/测试并训练
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=df['id_student']))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)

    # 5. 评估模型（默认阈值 0.5）
    # predict() 本身就是用阈值 0.5
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=== 默认阈值 = 0.5 ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")


    # 6. 保存模型
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    print(f"Model pipeline saved to {artifact_path}")


if __name__ == '__main__':
    main()
