# src/trainers/train_rf.py
"""
Train a Random Forest pipeline on the weekly data, evaluate and save the trained model.
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, recall_score, make_scorer
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

    # 3. 构建带 SMOTE 的 Pipeline（classifier 加 class_weight）
    pipeline = Pipeline([
        ('preproc', get_feature_pipeline(
            num_feats=num_feats, ord_feats=ord_feats, nom_feats=nom_feats, model_type='tree'
        )),
        ('smote',   SMOTE(random_state=random_state)),
        ('clf',     RandomForestClassifier(class_weight='balanced', random_state=random_state))
    ])

    # 4. 划分训练/测试并训练
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=test_size,
        random_state=random_state
    )
    pipeline.fit(X_train, y_train)

    # 5. 超参数搜索
    param_dist = {
        'clf__n_estimators': [100, 200, 300, 500],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 5],
        'clf__max_features': ['sqrt', 'log2', 0.3, 0.5],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring=make_scorer(recall_score),
        cv=StratifiedKFold(3, shuffle=True, random_state=random_state),
        verbose=2,
        n_jobs=-1,
        refit=True
    )
    search.fit(X_train, y_train)

    print("Best params:", search.best_params_)
    print(f"Best CV recall: {search.best_score_:.3f}")

    # 6. 用最优模型评估测试集
    best_pipe = search.best_estimator_
    y_proba = best_pipe.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.3).astype(int)  # 你也可以在这里调阈
    print("Test recall @ thresh=0.3:", recall_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # 7. 保存最优模型
    joblib.dump(best_pipe, artifact_path)
    print(f"Best pipeline saved to {artifact_path}")

if __name__ == '__main__':
    main()
