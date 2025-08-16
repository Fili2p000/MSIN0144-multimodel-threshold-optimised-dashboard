from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def get_feature_pipeline(
    num_feats,        # 普通连续数值
    ord_feats,        # 有序类别
    nom_feats,        # 无序类别
    high_card_feats=None,  # 高基数类别
    cyclic_feats=None,     # 周期／时间型特征
    log_feats=None,        # 需做 log1p 的偏态数值
    model_type='tree'
):
    """
    返回一个 ColumnTransformer，用于把 X[col_list] 转成模型可用的矩阵。

    参数：
      num_feats: List[str]，数值特征
      ord_feats: List[str]，有序类别
      nom_feats: List[str]，无序类别
      high_card_feats, cyclic_feats, log_feats: 可选的其他类型特征
      model_type: 'tree'、'knn'、'nb'
    """

    edu_order = [
        'No Formal quals', 'Lower Than A Level', 'A Level or Equivalent',
        'HE Qualification', 'Post Graduate Qualification'
    ]
    imd_order = [
        '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
        '50-60%', '60-70%', '70-80%', '80-90%', '90-100%', 'Unknown'
    ]
    age_order = ['0-35','35-55','55<=']

    # categories 参数接一个 list of lists，顺序要跟 ord_feats 对应
    ord_categories = [edu_order, imd_order, age_order]   



    # ====== 1) 根据模型类型，分别定义 num_pipe, ord_pipe, nom_pipe ======
    if model_type == 'tree':
        # 树模型：数值只填中位数；所有类别都用有序编码
        num_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median'))
        ])
        ord_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        nom_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

    elif model_type == 'knn':
        # KNN：数值填中位数后标准化；有序编码保持；无序做 One-Hot
        num_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale',  StandardScaler())
        ])
        ord_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        nom_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

    else:  # model_type == 'nb'
        # Naive Bayes：数值只填中位数；有序编码保持；无序做 One-Hot
        num_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median'))
        ])
        ord_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        nom_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

    # ====== 2) 其他可选特征类型的管道 ======
    transformers = [
        ('num', num_pipe, num_feats),
        ('ord', ord_pipe, ord_feats),
        ('nom', nom_pipe, nom_feats),
    ]

    if high_card_feats:
        # 高频类别：频次编码（也可换为 target encoding）
        freq_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('freq',   FunctionTransformer(lambda col: col.map(col.value_counts())))
        ])
        transformers.append(('hc', freq_pipe, high_card_feats))

    if cyclic_feats:
        # 周期特征：sin/cos 变换
        cyclic_pipe = Pipeline([
            ('cyc', FunctionTransformer(
                lambda arr: np.vstack([
                    np.sin(2*np.pi*arr / arr.max()),
                    np.cos(2*np.pi*arr / arr.max())
                ]).T,
                validate=False
            ))
        ])
        transformers.append(('cyc', cyclic_pipe, cyclic_feats))

    if log_feats:
        # 偏态数值：log1p → 填补 →（KNN 时再）标准化 
        steps = [('log1p', FunctionTransformer(np.log1p, validate=False)),
                 ('impute', SimpleImputer(strategy='median'))]
        if model_type == 'knn':
            steps.append(('scale', StandardScaler()))
        log_pipe = Pipeline(steps)
        transformers.append(('log', log_pipe, log_feats))

    # ====== 3) 组装 ColumnTransformer ======
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    return preprocessor
