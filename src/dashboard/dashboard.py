import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Dashboard Configuration
app = dash.Dash(__name__)
app.title = "Educational Analytics Dashboard - Assignment Failure Prediction"

def create_dummy_data():
    """Create dummy data for testing when actual data is not available"""
    print("Creating dummy data for testing...")
    np.random.seed(42)  # For reproducible results
    
    dummy_data = {
        'id_student': range(1, 201),
        'code_module': (['AAA'] * 50 + ['BBB'] * 50 + ['CCC'] * 50 + ['DDD'] * 50),
        'code_presentation': (['2013J'] * 100 + ['2014B'] * 100),
        'week': np.random.choice([1, 2, 3, 4, 5, 6], 200),
        'target_fail': np.random.choice([0, 1], 200, p=[0.7, 0.3]),
        'total_weekly_clicks': np.random.randint(0, 200, 200),
        'gender': np.random.choice(['M', 'F'], 200),
        'age_band': np.random.choice(['0-35', '35-55', '55<='], 200),
        'highest_education': np.random.choice(['HE Qualification', 'A Level or Equivalent', 'Lower Than A Level'], 200),
        'catboost_fail_probability': np.random.random(200),
        'xgboost_fail_probability': np.random.random(200),
        'random_forest_fail_probability': np.random.random(200),
        'logistic_regression_fail_probability': np.random.random(200),
        'ensemble_average_fail_probability': np.random.random(200),
        'ensemble_weighted_fail_probability': np.random.random(200),
        'ensemble_median_fail_probability': np.random.random(200)
    }
    return pd.DataFrame(dummy_data)

# Load data functions
def load_predictions():
    """Load the main predictions data"""
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '../../data/predicted/predictions_all_models.csv'),
        os.path.join(os.path.dirname(__file__), '../data/predicted/predictions_all_models.csv'),
        os.path.join(os.path.dirname(__file__), 'data/predicted/predictions_all_models.csv'),
        'data/predicted/predictions_all_models.csv',
        'predictions_all_models.csv'
    ]
    
    for data_path in possible_paths:
        try:
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                print(f"Predictions data loaded successfully from {data_path}! Shape: {df.shape}")
                return df
        except Exception as e:
            print(f"Error loading from {data_path}: {e}")
            continue
    
    print("Could not find predictions data file. Using dummy data for testing.")
    return create_dummy_data()

def load_student_info():
    """Load additional student information"""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '../../data/raw/anonymisedData/studentInfo.csv'),
        os.path.join(os.path.dirname(__file__), '../data/raw/anonymisedData/studentInfo.csv'),
        'data/raw/anonymisedData/studentInfo.csv'
    ]
    
    for data_path in possible_paths:
        try:
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                print(f"Student info loaded successfully from {data_path}! Shape: {df.shape}")
                return df
        except Exception as e:
            print(f"Error loading student info from {data_path}: {e}")
            continue
    
    print("Could not find student info file.")
    return pd.DataFrame()

def load_courses():
    """Load course information"""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '../../data/raw/anonymisedData/courses.csv'),
        os.path.join(os.path.dirname(__file__), '../data/raw/anonymisedData/courses.csv'),
        'data/raw/anonymisedData/courses.csv'
    ]
    
    for data_path in possible_paths:
        try:
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                print(f"Courses data loaded successfully from {data_path}! Shape: {df.shape}")
                return df
        except Exception as e:
            print(f"Error loading courses from {data_path}: {e}")
            continue
    
    print("Could not find courses file.")
    return pd.DataFrame()

def get_risk_level(probability):
    """Convert prediction probability to risk level"""
    if pd.isna(probability):
        return "Unknown", "#95a5a6"
    if probability >= 0.7:
        return "High Risk", "#e74c3c"
    elif probability >= 0.4:
        return "Medium Risk", "#f39c12"
    else:
        return "Low Risk", "#27ae60"

def get_available_courses(df):
    """Get list of available courses"""
    if not df.empty and 'code_module' in df.columns:
        courses = sorted(df['code_module'].unique().tolist())
        return [course for course in courses if pd.notna(course)]
    return []

def get_available_presentations(df, course=None):
    """Get list of available presentations for a course"""
    if not df.empty and 'code_presentation' in df.columns:
        if course:
            filtered_df = df[df['code_module'] == course]
            presentations = sorted(filtered_df['code_presentation'].unique().tolist())
            return [pres for pres in presentations if pd.notna(pres)]
        presentations = sorted(df['code_presentation'].unique().tolist())
        return [pres for pres in presentations if pd.notna(pres)]
    return []

def get_available_weeks(df):
    """Get list of available weeks"""
    if not df.empty and 'week' in df.columns:
        weeks = sorted(df['week'].unique().tolist())
        return [week for week in weeks if pd.notna(week)]
    return []

# Initialize data
print("=== Loading Data ===")
df_predictions = load_predictions()
df_student_info = load_student_info()
df_courses = load_courses()

print(f"Predictions data shape: {df_predictions.shape}")
print(f"Predictions data empty: {df_predictions.empty}")
if not df_predictions.empty:
    print(f"Predictions columns: {df_predictions.columns.tolist()}")
    print(f"Available courses: {get_available_courses(df_predictions)}")
print("=== Data Loading Complete ===")

# Model options
MODEL_OPTIONS = [
    {'label': 'CatBoost', 'value': 'catboost_fail_probability'},
    {'label': 'XGBoost', 'value': 'xgboost_fail_probability'},
    {'label': 'Random Forest', 'value': 'random_forest_fail_probability'},
    {'label': 'Logistic Regression', 'value': 'logistic_regression_fail_probability'},
    {'label': 'Ensemble (Average)', 'value': 'ensemble_average_fail_probability'},
    {'label': 'Ensemble (Weighted)', 'value': 'ensemble_weighted_fail_probability'},
    {'label': 'Ensemble (Median)', 'value': 'ensemble_median_fail_probability'}
]

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Educational Analytics Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.P("Assignment Failure Prediction - Multi-Model Early Warning System", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 18}),
        html.P("Based on Open University Learning Analytics Dataset", 
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': 14})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Tabs
    dcc.Tabs(id="main-tabs", value='student-analysis', children=[
        dcc.Tab(label='Student Risk Analysis', value='student-analysis'),
        dcc.Tab(label='Model Performance Analysis', value='model-performance')
    ]),
    
    # Tab content
    html.Div(id='tab-content')
])

# Student Analysis Tab Content
def create_student_analysis_tab():
    return html.Div([
        # Controls Section
        html.Div([
            html.Div([
                html.Label("Select Course:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='course-dropdown',
                    options=[{'label': course, 'value': course} for course in get_available_courses(df_predictions)],
                    value=get_available_courses(df_predictions)[0] if get_available_courses(df_predictions) else None,
                    style={'marginBottom': '10px'}
                )
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Presentation:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='presentation-dropdown',
                    style={'marginBottom': '10px'}
                )
            ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Select Week:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='week-cutoff-dropdown',
                    options=[{'label': f'Week {week}', 'value': week} for week in get_available_weeks(df_predictions)],
                    value=get_available_weeks(df_predictions)[-1] if get_available_weeks(df_predictions) else None,
                    style={'marginBottom': '10px'}
                )
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Select Model:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=MODEL_OPTIONS,
                    value='ensemble_weighted_fail_probability',
                    style={'marginBottom': '10px'}
                )
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Risk Filter:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='risk-filter-dropdown',
                    options=[
                        {'label': 'All Students', 'value': 'all'},
                        {'label': 'High Risk Only', 'value': 'high'},
                        {'label': 'Medium & High Risk', 'value': 'medium_high'},
                        {'label': 'Low Risk Only', 'value': 'low'}
                    ],
                    value='all',
                    style={'marginBottom': '10px'}
                )
            ], style={'width': '16%', 'display': 'inline-block'})
        ], style={'marginBottom': '30px'}),
        
        # Summary Cards
        html.Div([
            html.Div([
                html.H3("Total Students", style={'color': '#3498db', 'marginBottom': '5px'}),
                html.H2(id='total-students', children="0", style={'marginBottom': '5px'}),
                html.P(id='total-students-detail', children="", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '14%', 'display': 'inline-block', 'margin': '0.5%', 
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("High Risk", style={'color': '#e74c3c', 'marginBottom': '5px'}),
                html.H2(id='high-risk-count', children="0", style={'marginBottom': '5px'}),
                html.P(id='high-risk-percentage', children="", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '14%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Medium Risk", style={'color': '#f39c12', 'marginBottom': '5px'}),
                html.H2(id='medium-risk-count', children="0", style={'marginBottom': '5px'}),
                html.P(id='medium-risk-percentage', children="", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '14%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Low Risk", style={'color': '#27ae60', 'marginBottom': '5px'}),
                html.H2(id='low-risk-count', children="0", style={'marginBottom': '5px'}),
                html.P(id='low-risk-percentage', children="", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '14%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Avg Risk", style={'color': '#9b59b6', 'marginBottom': '5px'}),
                html.H2(id='avg-risk-score', children="0.000", style={'marginBottom': '5px'}),
                html.P("Class Average", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '14%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Actual Fails", style={'color': '#34495e', 'marginBottom': '5px'}),
                html.H2(id='actual-fails', children="0", style={'marginBottom': '5px'}),
                html.P(id='actual-fails-pct', children="", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '14%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
                                               
            html.Div([
                html.H3("Model Accuracy", style={'color': '#8e44ad', 'marginBottom': '5px'}),
                html.H2(id='model-accuracy', children="N/A", style={'marginBottom': '5px'}),
                html.P("Prediction vs Actual", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '14%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'})
        ], style={'marginBottom': '30px'}),
        
        # Main Content Area
        html.Div([
            # Student List Table
            html.Div([
                html.H3("Student Risk Assessment", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.P("Students are ranked by risk level. Click column headers to sort.", 
                       style={'color': '#7f8c8d', 'marginBottom': '10px'}),
                
                # 展开模型详情的按钮（只在Ensemble模型时显示）
                html.Div([
                    html.Button(
                        "Show Individual Model Predictions", 
                        id="expand-models-btn",
                        n_clicks=0,
                        style={
                            'display': 'none',  # 默认隐藏
                            'backgroundColor': '#3498db',
                            'color': 'white',
                            'border': 'none',
                            'padding': '8px 16px',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'marginBottom': '10px'
                        }
                    )
                ]),
                
                dash_table.DataTable(
                    id='student-table',
                    columns=[
                        {'name': 'Student ID', 'id': 'id_student', 'type': 'text'},
                        {'name': 'Risk Level', 'id': 'risk_level', 'type': 'text'},
                        {'name': 'Risk Probability', 'id': 'prediction_probability', 'type': 'numeric', 'format': {'specifier': '.3f'}},
                        # 隐藏的模型概率列
                        {'name': 'CatBoost', 'id': 'catboost_prob', 'type': 'numeric', 'format': {'specifier': '.3f'}, 'hideable': True},
                        {'name': 'XGBoost', 'id': 'xgboost_prob', 'type': 'numeric', 'format': {'specifier': '.3f'}, 'hideable': True},
                        {'name': 'Random Forest', 'id': 'random_forest_prob', 'type': 'numeric', 'format': {'specifier': '.3f'}, 'hideable': True},
                        {'name': 'Logistic Reg', 'id': 'logistic_regression_prob', 'type': 'numeric', 'format': {'specifier': '.3f'}, 'hideable': True},
                        {'name': 'Actual Result', 'id': 'target_fail', 'type': 'text'},
                        {'name': 'Week', 'id': 'week', 'type': 'numeric'},
                        {'name': 'Weekly Clicks', 'id': 'total_weekly_clicks', 'type': 'numeric'},
                        {'name': 'Gender', 'id': 'gender', 'type': 'text'},
                        {'name': 'Age', 'id': 'age_band', 'type': 'text'},
                        {'name': 'Education', 'id': 'highest_education', 'type': 'text'}
                    ],
                    hidden_columns=['catboost_prob', 'xgboost_prob', 'random_forest_prob', 'logistic_regression_prob'],  # 默认隐藏
                    data=[{
                        'id_student': 'Loading...',
                        'risk_level': 'Loading...',
                        'prediction_probability': 0.0,
                        'catboost_prob': 0.0,
                        'xgboost_prob': 0.0,
                        'random_forest_prob': 0.0,
                        'logistic_regression_prob': 0.0,
                        'target_fail': 'Loading...',
                        'week': 0,
                        'total_weekly_clicks': 0,
                        'gender': 'Loading...',
                        'age_band': 'Loading...',
                        'highest_education': 'Loading...'
                    }],
                    style_cell={
                        'textAlign': 'left', 
                        'padding': '10px',
                        'fontFamily': 'Arial',
                        'fontSize': '13px',
                        'maxWidth': '150px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis'
                    },
                    style_header={
                        'backgroundColor': '#34495e',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{risk_level} = "High Risk"'},
                            'backgroundColor': '#ffebee',
                            'color': 'black',
                            'border': '1px solid #ffcdd2'
                        },
                        {
                            'if': {'filter_query': '{risk_level} = "Medium Risk"'},
                            'backgroundColor': '#fff8e1',
                            'color': 'black',
                            'border': '1px solid #ffe0b2'
                        },
                        {
                            'if': {'filter_query': '{risk_level} = "Low Risk"'},
                            'backgroundColor': '#f1f8e9',
                            'color': 'black',
                            'border': '1px solid #c8e6c9'
                        },
                        {
                            'if': {'filter_query': '{target_fail} = "Failed"'},
                            'fontWeight': 'bold'
                        }
                    ],
                    sort_action="native",
                    filter_action="native",
                    page_size=15,
                    style_table={'overflowX': 'auto'}
                )
            ], style={'width': '100%', 'marginBottom': '30px'}),
            
            # Charts Section Row 1
            html.Div([
                html.Div([
                    html.H3("Risk Distribution", style={'color': '#2c3e50'}),
                    dcc.Graph(id='risk-distribution-chart')
                ], style={'width': '33%', 'display': 'inline-block', 'marginRight': '0.5%'}),
                
                html.Div([
                    html.H3("Model Performance", style={'color': '#2c3e50'}),
                    dcc.Graph(id='model-performance-chart')
                ], style={'width': '33%', 'display': 'inline-block', 'marginRight': '0.5%'}),
                
                html.Div([
                    html.H3("Actual vs Predicted", style={'color': '#2c3e50'}),
                    dcc.Graph(id='actual-vs-predicted-chart')
                ], style={'width': '33%', 'display': 'inline-block'})
            ]),
            
            # Charts Section Row 2
            html.Div([
                html.Div([
                    html.H3("Weekly Risk Progression", style={'color': '#2c3e50'}),
                    dcc.Graph(id='weekly-progression-chart')
                ], style={'width': '50%', 'display': 'inline-block', 'marginRight': '0%'}),
                
                html.Div([
                    html.H3("VLE Activity vs Risk", style={'color': '#2c3e50'}),
                    dcc.Graph(id='vle-risk-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ], style={'marginTop': '30px'}),
            
            # Model Comparison Section
            html.Div([
                html.H3("Multi-Model Comparison", style={'color': '#2c3e50'}),
                dcc.Graph(id='model-comparison-chart')
            ], style={'marginTop': '30px'})
        ])
    ])

# Model Performance Analysis Tab Content
def create_model_performance_tab():
    return html.Div([
        html.H2("Model Performance Analysis - Full Dataset", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        
        # Overall Performance Summary Cards
        html.Div([
            html.Div([
                html.H3("Total Samples", style={'color': '#3498db', 'marginBottom': '5px'}),
                html.H2(id='total-samples', children="0", style={'marginBottom': '5px'}),
                html.P("Complete Dataset", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '24%', 'display': 'inline-block', 'margin': '0.5%', 
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Best Model", style={'color': '#27ae60', 'marginBottom': '5px'}),
                html.H2(id='best-model', children="N/A", style={'marginBottom': '5px'}),
                html.P(id='best-model-score', children="", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '24%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Fail Rate", style={'color': '#e74c3c', 'marginBottom': '5px'}),
                html.H2(id='overall-fail-rate', children="0%", style={'marginBottom': '5px'}),
                html.P("Actual Failure Rate", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '24%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'}),
            
            html.Div([
                html.H3("Avg Precision", style={'color': '#9b59b6', 'marginBottom': '5px'}),
                html.H2(id='avg-precision', children="0.000", style={'marginBottom': '5px'}),
                html.P("Across All Models", style={'fontSize': '12px', 'color': '#7f8c8d'})
            ], className='summary-card', style={'width': '24%', 'display': 'inline-block', 'margin': '0.5%',
                                               'padding': '15px', 'backgroundColor': '#ecf0f1', 'textAlign': 'center'})
        ], style={'marginBottom': '30px'}),
        
        # Performance Metrics Table
        html.Div([
            html.H3("Model Performance Metrics", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            dash_table.DataTable(
                id='performance-metrics-table',
                columns=[
                    {'name': 'Model', 'id': 'model', 'type': 'text'},
                    {'name': 'Accuracy', 'id': 'accuracy', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                    {'name': 'Precision (Pass)', 'id': 'precision_0', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                    {'name': 'Recall (Pass)', 'id': 'recall_0', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                    {'name': 'F1 (Pass)', 'id': 'f1_0', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                    {'name': 'Precision (Fail)', 'id': 'precision_1', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                    {'name': 'Recall (Fail)', 'id': 'recall_1', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                    {'name': 'F1 (Fail)', 'id': 'f1_1', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                    {'name': 'AUC-PR', 'id': 'auc_pr', 'type': 'numeric', 'format': {'specifier': '.4f'}}
                ],
                data=[],
                style_cell={
                    'textAlign': 'center', 
                    'padding': '10px',
                    'fontFamily': 'Arial',
                    'fontSize': '13px'
                },
                style_header={
                    'backgroundColor': '#34495e',
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f8f9fa'
                    }
                ]
            )
        ], style={'width': '100%', 'marginBottom': '30px'}),
        
        # Visualization Charts
        html.Div([
            # Row 1: Confusion Matrices
            html.Div([
                html.H3("Confusion Matrices", style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '20px'}),
                dcc.Graph(id='confusion-matrices-chart')
            ], style={'width': '100%', 'marginBottom': '30px'}),
            
            # Row 2: Performance Comparison Charts
            html.Div([
                html.Div([
                    html.H3("F1 Score Comparison", style={'color': '#2c3e50'}),
                    dcc.Graph(id='f1-comparison-chart')
                ], style={'width': '50%', 'display': 'inline-block', 'marginRight': '0%'}),
                
                html.Div([
                    html.H3("Precision-Recall Comparison", style={'color': '#2c3e50'}),
                    dcc.Graph(id='precision-recall-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ], style={'marginBottom': '30px'}),
            
            # Row 3: AUC-PR Curves
            html.Div([
                html.H3("Precision-Recall Curves", style={'color': '#2c3e50'}),
                dcc.Graph(id='auc-pr-curves-chart')
            ], style={'width': '100%', 'marginBottom': '30px'})
        ])
    ])

# Tab content callback
@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_content(tab):
    if tab == 'student-analysis':
        return create_student_analysis_tab()
    elif tab == 'model-performance':
        return create_model_performance_tab()

# Performance analysis callback
@app.callback(
    [Output('total-samples', 'children'),
     Output('best-model', 'children'),
     Output('best-model-score', 'children'),
     Output('overall-fail-rate', 'children'),
     Output('avg-precision', 'children'),
     Output('performance-metrics-table', 'data'),
     Output('confusion-matrices-chart', 'figure'),
     Output('f1-comparison-chart', 'figure'),
     Output('precision-recall-chart', 'figure'),
     Output('auc-pr-curves-chart', 'figure')],
    Input('main-tabs', 'value')
)
def update_performance_analysis(tab):
    if tab != 'model-performance' or df_predictions.empty:
        empty_fig = go.Figure()
        return ("0", "N/A", "", "0%", "0.000", [], empty_fig, empty_fig, empty_fig, empty_fig)
    
    try:
        print("Calculating model performance metrics for full dataset...")
        
        # Get the latest week data for each student to avoid duplicates
        latest_data = df_predictions.groupby('id_student').last().reset_index()
        
        # Define model columns
        model_columns = [
            'catboost_fail_probability',
            'xgboost_fail_probability', 
            'random_forest_fail_probability',
            'logistic_regression_fail_probability'
        ]
        
        # Filter out missing model columns
        available_models = [col for col in model_columns if col in latest_data.columns]
        
        if not available_models or 'target_fail' not in latest_data.columns:
            print("Required model columns or target not found")
            empty_fig = go.Figure()
            return ("0", "N/A", "", "0%", "0.000", [], empty_fig, empty_fig, empty_fig, empty_fig)
        
        print(f"Found {len(available_models)} models: {available_models}")
        
        # Remove rows with missing target values
        clean_data = latest_data.dropna(subset=['target_fail'] + available_models).copy()
        
        if clean_data.empty:
            print("No clean data available")
            empty_fig = go.Figure()
            return ("0", "N/A", "", "0%", "0.000", [], empty_fig, empty_fig, empty_fig, empty_fig)
        
        total_samples = len(clean_data)
        y_true = clean_data['target_fail'].astype(int)
        fail_rate = y_true.mean()
        
        print(f"Total samples: {total_samples}, Fail rate: {fail_rate:.3f}")
        
        # Calculate metrics for each model
        performance_data = []
        confusion_matrices = {}
        auc_pr_data = {}
        best_f1 = 0
        best_model_name = "N/A"
        
        for model_col in available_models:
            model_name = model_col.replace('_fail_probability', '').replace('_', ' ').title()
            y_pred_prob = clean_data[model_col]
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            try:
                # Calculate basic metrics
                accuracy = accuracy_score(y_true, y_pred)
                
                # Calculate per-class metrics
                precision = precision_score(y_true, y_pred, average=None, zero_division=0)
                recall = recall_score(y_true, y_pred, average=None, zero_division=0)
                f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
                
                # Ensure we have metrics for both classes
                if len(precision) >= 2 and len(recall) >= 2 and len(f1) >= 2:
                    precision_0, precision_1 = precision[0], precision[1]
                    recall_0, recall_1 = recall[0], recall[1]
                    f1_0, f1_1 = f1[0], f1[1]
                else:
                    precision_0 = precision_1 = recall_0 = recall_1 = f1_0 = f1_1 = 0
                
                # Calculate AUC-PR
                try:
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
                    auc_pr = auc(recall_curve, precision_curve)
                    auc_pr_data[model_name] = {
                        'precision': precision_curve,
                        'recall': recall_curve,
                        'auc': auc_pr
                    }
                except Exception as e:
                    print(f"Error calculating AUC-PR for {model_name}: {e}")
                    auc_pr = 0
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                confusion_matrices[model_name] = cm
                
                # Track best model
                avg_f1 = (f1_0 + f1_1) / 2
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_model_name = model_name
                
                performance_data.append({
                    'model': model_name,
                    'accuracy': accuracy,
                    'precision_0': precision_0,
                    'recall_0': recall_0,
                    'f1_0': f1_0,
                    'precision_1': precision_1,
                    'recall_1': recall_1,
                    'f1_1': f1_1,
                    'auc_pr': auc_pr
                })
                
                print(f"{model_name}: Accuracy={accuracy:.3f}, F1_avg={avg_f1:.3f}, AUC-PR={auc_pr:.3f}")
                
            except Exception as e:
                print(f"Error calculating metrics for {model_name}: {e}")
                continue
        
        # Calculate average precision across all models
        if performance_data:
            avg_precision = np.mean([row['precision_1'] for row in performance_data])
        else:
            avg_precision = 0
        
        # Create visualizations
        # 1. Confusion Matrices
        n_models = len(confusion_matrices)
        if n_models > 0:
            fig_cm = make_subplots(
                rows=1, cols=n_models,
                subplot_titles=list(confusion_matrices.keys()),
                specs=[[{'type': 'heatmap'}] * n_models]
            )
            
            for i, (model_name, cm) in enumerate(confusion_matrices.items()):
                fig_cm.add_trace(
                    go.Heatmap(
                        z=cm,
                        x=['Pass', 'Fail'],
                        y=['Pass', 'Fail'],
                        colorscale='Blues',
                        showscale=(i == 0),
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 16}
                    ),
                    row=1, col=i+1
                )
            
            fig_cm.update_layout(
                title="Confusion Matrices - All Models",
                height=400
            )
        else:
            fig_cm = go.Figure()
        
        # 2. F1 Score Comparison
        if performance_data:
            models = [row['model'] for row in performance_data]
            f1_pass = [row['f1_0'] for row in performance_data]
            f1_fail = [row['f1_1'] for row in performance_data]
            
            fig_f1 = go.Figure(data=[
                go.Bar(name='F1 (Pass)', x=models, y=f1_pass, marker_color='lightblue'),
                go.Bar(name='F1 (Fail)', x=models, y=f1_fail, marker_color='lightcoral')
            ])
            fig_f1.update_layout(
                title="F1 Score Comparison by Model and Class",
                barmode='group',
                xaxis_title="Model",
                yaxis_title="F1 Score"
            )
        else:
            fig_f1 = go.Figure()
        
        # 3. Precision-Recall Comparison
        if performance_data:
            fig_pr = go.Figure()
            
            # Add scatter points for each model
            precisions = [row['precision_1'] for row in performance_data]
            recalls = [row['recall_1'] for row in performance_data]
            
            fig_pr.add_trace(go.Scatter(
                x=recalls,
                y=precisions,
                mode='markers+text',
                text=models,
                textposition="top center",
                marker=dict(size=10, color='red'),
                name='Models'
            ))
            
            fig_pr.update_layout(
                title="Precision vs Recall (Fail Class)",
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
        else:
            fig_pr = go.Figure()
        
        # 4. AUC-PR Curves
        if auc_pr_data:
            fig_auc = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (model_name, data) in enumerate(auc_pr_data.items()):
                fig_auc.add_trace(go.Scatter(
                    x=data['recall'],
                    y=data['precision'],
                    mode='lines',
                    name=f"{model_name} (AUC={data['auc']:.3f})",
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig_auc.update_layout(
                title="Precision-Recall Curves",
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
        else:
            fig_auc = go.Figure()
        
        return (
            f"{total_samples:,}",
            best_model_name,
            f"F1: {best_f1:.3f}",
            f"{fail_rate*100:.1f}%",
            f"{avg_precision:.3f}",
            performance_data,
            fig_cm, fig_f1, fig_pr, fig_auc
        )
        
    except Exception as e:
        print(f"Error in performance analysis: {e}")
        traceback.print_exc()
        empty_fig = go.Figure()
        return ("0", "N/A", "", "0%", "0.000", [], empty_fig, empty_fig, empty_fig, empty_fig)
@app.callback(
    Output('expand-models-btn', 'style'),
    Output('expand-models-btn', 'children'),
    Input('model-dropdown', 'value'),
    Input('expand-models-btn', 'n_clicks')
)
def control_expand_button(selected_model, n_clicks):
    # 只在选择Ensemble模型时显示按钮
    if selected_model and 'ensemble' in selected_model.lower():
        # 根据点击次数决定按钮文本
        if n_clicks % 2 == 0:
            button_text = "Show Individual Model Predictions"
        else:
            button_text = "Hide Individual Model Predictions"
        
        button_style = {
            'display': 'inline-block',
            'backgroundColor': '#3498db',
            'color': 'white',
            'border': 'none',
            'padding': '8px 16px',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'marginBottom': '10px'
        }
        return button_style, button_text
    else:
        # 非Ensemble模型时隐藏按钮
        button_style = {'display': 'none'}
        return button_style, "Show Individual Model Predictions"

# Callback to control table column visibility
@app.callback(
    Output('student-table', 'hidden_columns'),
    Input('model-dropdown', 'value'),
    Input('expand-models-btn', 'n_clicks')
)
def control_table_columns(selected_model, n_clicks):
    model_columns = ['catboost_prob', 'xgboost_prob', 'random_forest_prob', 'logistic_regression_prob']
    
    # 如果不是Ensemble模型，始终隐藏所有模型列
    if not selected_model or 'ensemble' not in selected_model.lower():
        return model_columns
    
    # 如果是Ensemble模型，根据按钮点击状态决定是否显示
    if n_clicks % 2 == 1:  # 奇数次点击表示展开
        return []  # 显示所有列
    else:  # 偶数次点击表示收起
        return model_columns  # 隐藏模型列
@app.callback(
    Output('presentation-dropdown', 'options'),
    Output('presentation-dropdown', 'value'),
    Input('course-dropdown', 'value')
)
def update_presentation_dropdown(selected_course):
    print(f"Updating presentation dropdown for course: {selected_course}")
    if selected_course and not df_predictions.empty:
        presentations = get_available_presentations(df_predictions, selected_course)
        options = [{'label': pres, 'value': pres} for pres in presentations]
        value = presentations[0] if presentations else None
        print(f"Found presentations: {presentations}")
        return options, value
    return [], None

# Main callback
@app.callback(
    [Output('total-students', 'children'),
     Output('total-students-detail', 'children'),
     Output('high-risk-count', 'children'),
     Output('high-risk-percentage', 'children'),
     Output('medium-risk-count', 'children'),
     Output('medium-risk-percentage', 'children'),
     Output('low-risk-count', 'children'),
     Output('low-risk-percentage', 'children'),
     Output('avg-risk-score', 'children'),
     Output('actual-fails', 'children'),
     Output('actual-fails-pct', 'children'),
     Output('model-accuracy', 'children'),
     Output('student-table', 'data'),
     Output('risk-distribution-chart', 'figure'),
     Output('model-performance-chart', 'figure'),
     Output('actual-vs-predicted-chart', 'figure'),
     Output('weekly-progression-chart', 'figure'),
     Output('vle-risk-chart', 'figure'),
     Output('model-comparison-chart', 'figure')],
    [Input('course-dropdown', 'value'),
     Input('presentation-dropdown', 'value'),
     Input('week-cutoff-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('risk-filter-dropdown', 'value')]
)
def update_dashboard(selected_course, selected_presentation, week_cutoff, selected_model, risk_filter):
    print(f"=== Callback triggered ===")
    print(f"Inputs: course={selected_course}, presentation={selected_presentation}, week={week_cutoff}, model={selected_model}, filter={risk_filter}")
    
    # 创建默认返回值
    def get_default_return():
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available or data loading failed",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        
        default_table_data = [{
            'id_student': 'No Data Available',
            'risk_level': 'Check Data Source',
            'prediction_probability': 0.0,
            'catboost_prob': 0.0,
            'xgboost_prob': 0.0,
            'random_forest_prob': 0.0,
            'logistic_regression_prob': 0.0,
            'target_fail': 'No Data',
            'week': 0,
            'total_weekly_clicks': 0,
            'gender': 'No Data',
            'age_band': 'No Data',
            'highest_education': 'No Data'
        }]
        
        return ("0", "No data available", "0", "(0%)", "0", "(0%)", "0", "(0%)", 
                "0.000", "0", "(0%)", "N/A", default_table_data,
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig)
    
    # 检查基本条件
    if df_predictions.empty:
        print("ERROR: Predictions data is empty!")
        return get_default_return()
    
    if not all([selected_course, selected_presentation, week_cutoff, selected_model]):
        print(f"ERROR: Some inputs are None! course={selected_course}, presentation={selected_presentation}, week={week_cutoff}, model={selected_model}")
        return get_default_return()
    
    # 检查选择的模型列是否存在
    if selected_model not in df_predictions.columns:
        print(f"ERROR: Selected model {selected_model} not found in data columns!")
        print(f"Available columns: {df_predictions.columns.tolist()}")
        return get_default_return()
    
    try:
        # 过滤数据 - 只显示选定的那一周
        print(f"Filtering data for week {week_cutoff} only...")
        filtered_df = df_predictions[
            (df_predictions['code_module'] == selected_course) & 
            (df_predictions['code_presentation'] == selected_presentation) &
            (df_predictions['week'] == week_cutoff)  # 改为只显示选定的周
        ].copy()
        
        print(f"Filtered data shape: {filtered_df.shape}")
        
        if filtered_df.empty:
            print("ERROR: No data after filtering!")
            return get_default_return()
        
        # 由于现在只显示一周的数据，不需要获取"最新"数据，直接使用过滤后的数据
        print("Using single week data...")
        current_risk_df = filtered_df.copy()
        print(f"Current risk data shape: {current_risk_df.shape}")
        print(f"Available columns: {current_risk_df.columns.tolist()}")
        
        # 添加预测概率和风险等级
        print("Adding risk levels...")
        current_risk_df['prediction_probability'] = current_risk_df[selected_model].fillna(0)
        
        # 安全地应用风险等级函数
        risk_data = current_risk_df['prediction_probability'].apply(get_risk_level)
        current_risk_df['risk_level'] = [item[0] for item in risk_data]
        current_risk_df['risk_color'] = [item[1] for item in risk_data]
        
        # 处理目标变量
        if 'target_fail' in current_risk_df.columns:
            current_risk_df['target_fail_text'] = current_risk_df['target_fail'].map({0: 'Pass', 1: 'Failed'}).fillna('Unknown')
        else:
            current_risk_df['target_fail_text'] = 'Unknown'
            current_risk_df['target_fail'] = 0
        
        print(f"Sample data after processing:")
        if not current_risk_df.empty:
            sample_cols = ['id_student', 'prediction_probability', 'risk_level', 'target_fail']
            available_cols = [col for col in sample_cols if col in current_risk_df.columns]
            print(current_risk_df[available_cols].head())
        
        # 应用风险过滤
        original_count = len(current_risk_df)
        if risk_filter == 'high':
            current_risk_df = current_risk_df[current_risk_df['risk_level'] == 'High Risk']
        elif risk_filter == 'medium_high':
            current_risk_df = current_risk_df[current_risk_df['risk_level'].isin(['High Risk', 'Medium Risk'])]
        elif risk_filter == 'low':
            current_risk_df = current_risk_df[current_risk_df['risk_level'] == 'Low Risk']
        
        print(f"After risk filter: {len(current_risk_df)} students (was {original_count})")
        
        if current_risk_df.empty:
            print("ERROR: No data after risk filtering!")
            return get_default_return()
        
        # 计算统计信息
        print("Calculating statistics...")
        total_students = len(current_risk_df)
        high_risk = len(current_risk_df[current_risk_df['risk_level'] == 'High Risk'])
        medium_risk = len(current_risk_df[current_risk_df['risk_level'] == 'Medium Risk'])
        low_risk = len(current_risk_df[current_risk_df['risk_level'] == 'Low Risk'])
        
        avg_risk = current_risk_df['prediction_probability'].mean()
        actual_fails = current_risk_df['target_fail'].sum()
        
        # 计算百分比
        high_risk_pct = f"({high_risk/total_students*100:.1f}%)"
        medium_risk_pct = f"({medium_risk/total_students*100:.1f}%)"
        low_risk_pct = f"({low_risk/total_students*100:.1f}%)"
        actual_fails_pct = f"({actual_fails/total_students*100:.1f}%)"
        
        # 计算准确率
        predicted_labels = (current_risk_df['prediction_probability'] >= 0.5).astype(int)
        actual_labels = current_risk_df['target_fail']
        accuracy = (predicted_labels == actual_labels).sum() / len(current_risk_df)
        accuracy_str = f"{accuracy:.3f}"
        
        print(f"Statistics: total={total_students}, high={high_risk}, medium={medium_risk}, low={low_risk}")
        
        # 准备表格数据
        print("Preparing table data...")
        table_data = []
        table_df = current_risk_df.sort_values(['prediction_probability'], ascending=False).head(100)
        
        for _, row in table_df.iterrows():
            try:
                record = {
                    'id_student': str(int(row['id_student'])) if pd.notna(row['id_student']) else 'N/A',
                    'risk_level': str(row['risk_level']),
                    'prediction_probability': round(float(row['prediction_probability']), 4),
                    # 添加所有模型的概率数据
                    'catboost_prob': round(float(row.get('catboost_fail_probability', 0)), 4),
                    'xgboost_prob': round(float(row.get('xgboost_fail_probability', 0)), 4),
                    'random_forest_prob': round(float(row.get('random_forest_fail_probability', 0)), 4),
                    'logistic_regression_prob': round(float(row.get('logistic_regression_fail_probability', 0)), 4),
                    'target_fail': str(row['target_fail_text']),
                    'week': int(row['week']) if pd.notna(row['week']) else 0,
                    'total_weekly_clicks': int(row['total_weekly_clicks']) if pd.notna(row['total_weekly_clicks']) else 0,
                    'gender': str(row.get('gender', 'N/A')),
                    'age_band': str(row.get('age_band', 'N/A')),
                    'highest_education': str(row.get('highest_education', 'N/A'))[:30]
                }
                table_data.append(record)
            except Exception as e:
                print(f"Error processing table row: {e}")
                continue
        
        print(f"Created {len(table_data)} table records")
        if table_data:
            print(f"First record sample: {table_data[0]}")
        
        # 如果没有表格数据，创建一个默认记录
        if not table_data:
            table_data = [{
                'id_student': 'No Data',
                'risk_level': 'No Data',
                'prediction_probability': 0.0,
                'catboost_prob': 0.0,
                'xgboost_prob': 0.0,
                'random_forest_prob': 0.0,
                'logistic_regression_prob': 0.0,
                'target_fail': 'No Data',
                'week': 0,
                'total_weekly_clicks': 0,
                'gender': 'No Data',
                'age_band': 'No Data',
                'highest_education': 'No Data'
            }]
        
        # 创建图表
        print("Creating charts...")
        
        # 1. 风险分布图
        risk_counts = current_risk_df['risk_level'].value_counts()
        if not risk_counts.empty:
            risk_dist_fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title=f"Risk Distribution - {selected_course} {selected_presentation} Week {week_cutoff}",
                color_discrete_map={'High Risk': '#e74c3c', 'Medium Risk': '#f39c12', 'Low Risk': '#27ae60'}
            )
        else:
            risk_dist_fig = go.Figure()
            risk_dist_fig.add_annotation(text="No risk data available", 
                                       xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # 2. 模型性能图（混淆矩阵）
        try:
            predicted_risk = (current_risk_df['prediction_probability'] >= 0.5).astype(int)
            actual_risk = current_risk_df['target_fail']
            
            tp = ((predicted_risk == 1) & (actual_risk == 1)).sum()
            fp = ((predicted_risk == 1) & (actual_risk == 0)).sum()
            tn = ((predicted_risk == 0) & (actual_risk == 0)).sum()
            fn = ((predicted_risk == 0) & (actual_risk == 1)).sum()
            
            confusion_matrix = [[tn, fp], [fn, tp]]
            model_perf_fig = px.imshow(
                confusion_matrix,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Pass', 'Fail'],
                y=['Pass', 'Fail'],
                title=f"Model Performance Week {week_cutoff} - {selected_model.replace('_', ' ').title()}"
            )
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            model_perf_fig = go.Figure()
            model_perf_fig.add_annotation(text="Error creating performance chart", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # 3. 实际vs预测散点图
        try:
            actual_vs_pred_fig = px.scatter(
                current_risk_df,
                x='target_fail',
                y='prediction_probability',
                color='risk_level',
                title=f"Actual Results vs Predicted Risk - Week {week_cutoff}",
                labels={'target_fail': 'Actual Result (0=Pass, 1=Fail)', 'prediction_probability': 'Predicted Risk'},
                color_discrete_map={'High Risk': '#e74c3c', 'Medium Risk': '#f39c12', 'Low Risk': '#27ae60'}
            )
        except Exception as e:
            print(f"Error creating actual vs predicted chart: {e}")
            actual_vs_pred_fig = go.Figure()
            actual_vs_pred_fig.add_annotation(text="Error creating scatter plot", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # 4. 周度风险进展图 (需要使用所有周的数据，而不是只用当前选定的周)
        try:
            # 为了显示progression，我们需要获取所有周的数据
            all_weeks_df = df_predictions[
                (df_predictions['code_module'] == selected_course) & 
                (df_predictions['code_presentation'] == selected_presentation)
            ]
            
            weeks = sorted(all_weeks_df['week'].unique())
            weekly_stats = []
            for week in weeks:
                week_data = all_weeks_df[all_weeks_df['week'] == week]
                if not week_data.empty:
                    week_predictions = week_data[selected_model]
                    high_risk_count = sum(week_predictions >= 0.7)
                    medium_risk_count = sum((week_predictions >= 0.4) & (week_predictions < 0.7))
                    low_risk_count = sum(week_predictions < 0.4)
                    total = len(week_predictions)
                    
                    if total > 0:
                        weekly_stats.extend([
                            {'Week': week, 'Risk Level': 'High Risk', 'Percentage': high_risk_count/total*100},
                            {'Week': week, 'Risk Level': 'Medium Risk', 'Percentage': medium_risk_count/total*100},
                            {'Week': week, 'Risk Level': 'Low Risk', 'Percentage': low_risk_count/total*100}
                        ])
            
            if weekly_stats:
                weekly_df = pd.DataFrame(weekly_stats)
                weekly_fig = px.line(
                    weekly_df, x='Week', y='Percentage', color='Risk Level',
                    title=f"Weekly Risk Progression (All Weeks)",
                    color_discrete_map={'High Risk': '#e74c3c', 'Medium Risk': '#f39c12', 'Low Risk': '#27ae60'}
                )
                # 高亮显示当前选定的周
                weekly_fig.add_vline(x=week_cutoff, line_dash="dash", line_color="red", 
                                   annotation_text=f"Selected Week {week_cutoff}")
            else:
                weekly_fig = go.Figure()
                weekly_fig.add_annotation(text="No weekly data available",
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        except Exception as e:
            print(f"Error creating weekly progression chart: {e}")
            weekly_fig = go.Figure()
            weekly_fig.add_annotation(text="Error creating weekly chart",
                                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # 5. VLE活动vs风险图
        try:
            vle_fig = px.scatter(
                current_risk_df,
                x='total_weekly_clicks',
                y='prediction_probability',
                color='risk_level',
                title=f"VLE Activity vs Assignment Failure Risk - Week {week_cutoff}",
                labels={'total_weekly_clicks': 'Weekly VLE Clicks', 'prediction_probability': 'Failure Risk'},
                color_discrete_map={'High Risk': '#e74c3c', 'Medium Risk': '#f39c12', 'Low Risk': '#27ae60'}
            )
        except Exception as e:
            print(f"Error creating VLE chart: {e}")
            vle_fig = go.Figure()
            vle_fig.add_annotation(text="Error creating VLE chart",
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # 6. 模型比较图
        try:
            model_cols = [col for col in current_risk_df.columns if 'fail_probability' in col]
            if len(model_cols) > 1:
                model_comparison_data = []
                for model in model_cols:
                    model_name = model.replace('_fail_probability', '').replace('_', ' ').title()
                    avg_pred = current_risk_df[model].mean()
                    model_comparison_data.append({'Model': model_name, 'Average Prediction': avg_pred})
                
                comparison_df = pd.DataFrame(model_comparison_data)
                model_comp_fig = px.bar(
                    comparison_df,
                    x='Model',
                    y='Average Prediction',
                    title=f"Average Risk Predictions by Model - Week {week_cutoff}"
                )
                model_comp_fig.update_xaxes(tickangle=45)
            else:
                model_comp_fig = go.Figure()
                model_comp_fig.add_annotation(text="Not enough models for comparison",
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        except Exception as e:
            print(f"Error creating model comparison chart: {e}")
            model_comp_fig = go.Figure()
            model_comp_fig.add_annotation(text="Error creating model comparison",
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        print(f"Returning successful result with {len(table_data)} table records")
        return (
            str(total_students), f"Week {week_cutoff}",
            str(high_risk), high_risk_pct,
            str(medium_risk), medium_risk_pct,
            str(low_risk), low_risk_pct,
            f"{avg_risk:.3f}",
            str(int(actual_fails)), actual_fails_pct,
            accuracy_str,
            table_data,
            risk_dist_fig, model_perf_fig, actual_vs_pred_fig,
            weekly_fig, vle_fig, model_comp_fig
        )
        
    except Exception as e:
        print(f"CRITICAL ERROR in callback: {e}")
        traceback.print_exc()
        return get_default_return()

if __name__ == '__main__':
    print("Starting dashboard...")
    app.run(debug=True, host='127.0.0.1', port=8050)