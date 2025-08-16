import os
import pandas as pd
import numpy as np
# ----------------------------------------------------------------------------
# data/raw/anonymisedData/
#   ├── assessments.csv
#   ├── courses.csv
#   ├── studentAssessment.csv
#   ├── studentInfo.csv
#   ├── studentRegistration.csv
#   ├── studentVle.csv
#   └── vle.csv
# ----------------------------------------------------------------------------

FILENAMES = [
    'assessments.csv',
    'courses.csv',
    'studentAssessment.csv',
    'studentInfo.csv',
    'studentRegistration.csv',
    'studentVle.csv',
    'vle.csv'
]

def add_engineered_features(train_weekly: pd.DataFrame, raw: dict) -> pd.DataFrame:
    """
    Call this function at the end of build_weekly_dataset to add more engineered features
    """
    df = train_weekly.copy()
    
    # === 1. Course-wise cumulative features ===(Aborted due to performance issues)
    def add_course_cumulative_features(df):
        # Calculate cumulative statistics for each student in each course
        activity_cols = [col for col in df.columns if col.startswith('sum_click_')]
        
        # Sort dataframe first to ensure proper ordering
        df = df.sort_values(['id_student', 'code_module', 'week']).reset_index(drop=True)
        
        for col in activity_cols:
            # Cumulative total clicks (cum_click series already exists, this calculates the sum)
            df[f'total_{col}_to_date'] = df.groupby(['id_student', 'code_module'])[col].cumsum()
            
            # Average clicks per week
            df[f'avg_{col}_to_date'] = df[f'total_{col}_to_date'] / (df['week'] + 1)
            
            # Rolling 3-week moving average (if enough weeks available)
            # Fix the index issue by using transform instead
            df[f'{col}_ma3'] = df.groupby(['id_student', 'code_module'])[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            
            # Learning consistency (coefficient of variation)
            # Fix the index issue by using transform instead
            rolling_std = df.groupby(['id_student', 'code_module'])[col].transform(
                lambda x: x.expanding().std()
            )
            rolling_mean = df.groupby(['id_student', 'code_module'])[col].transform(
                lambda x: x.expanding().mean()
            )
            df[f'{col}_consistency'] = rolling_std / (rolling_mean + 1e-8)  # Avoid division by zero
        
        return df
    
    # === 2. Cross-course comparison features ===
    def add_cross_course_features(df):
        # Calculate total weekly activity
        activity_cols = [col for col in df.columns if col.startswith('sum_click_')]
        df['total_weekly_clicks'] = df[activity_cols].sum(axis=1)
        
        # Total clicks across all courses per week
        weekly_totals = df.groupby(['id_student', 'week'])['total_weekly_clicks'].sum().reset_index()
        weekly_totals.rename(columns={'total_weekly_clicks': 'all_courses_clicks_this_week'}, inplace=True)
        df = df.merge(weekly_totals, on=['id_student', 'week'], how='left')
        
        # Relative effort ratio for current course
        df['relative_effort_ratio'] = df['total_weekly_clicks'] / (df['all_courses_clicks_this_week'] + 1e-8)
        
        # Number of courses the student is taking this week
        course_counts = df.groupby(['id_student', 'week']).size().reset_index(name='courses_count_this_week')
        df = df.merge(course_counts, on=['id_student', 'week'], how='left')
        
        # Course priority ranking
        df['course_priority_rank'] = df.groupby(['id_student', 'week'])['total_weekly_clicks'].rank(method='dense', ascending=False)
        
        return df
    
    # === 3. Temporal features ===
    def add_temporal_features(df, raw):
        # Merge course length information
        courses = raw['courses'][['code_module', 'code_presentation', 'module_presentation_length']]
        df = df.merge(courses, on=['code_module', 'code_presentation'], how='left')
        
        # Course progress ratio
        df['course_progress_ratio'] = df['week'] / (df['module_presentation_length'] / 7)
        df['course_progress_ratio'] = df['course_progress_ratio'].clip(0, 1)  # Limit to 0-1 range
        
        # Weeks remaining until course end
        df['weeks_to_course_end'] = (df['module_presentation_length'] / 7) - df['week']
        df['weeks_to_course_end'] = df['weeks_to_course_end'].clip(0, None)  # Cannot be negative
        
        # Course stage indicators
        df['is_early_stage'] = (df['course_progress_ratio'] <= 0.25).astype(int)
        df['is_mid_stage'] = ((df['course_progress_ratio'] > 0.25) & (df['course_progress_ratio'] <= 0.75)).astype(int)
        df['is_final_stage'] = (df['course_progress_ratio'] > 0.75).astype(int)
        
        return df
    
    # === 4. Learning pattern features ===(Aborted due to performance issues)
    def add_learning_pattern_features(df):
        # Learning activity trends
        activity_cols = [col for col in df.columns if col.startswith('sum_click_')]
        
        for col in activity_cols:
            # Change from previous week
            df[f'{col}_change_from_prev'] = df.groupby(['id_student', 'code_module'])[col].diff()
            
            # Deviation from personal average
            personal_avg = df.groupby(['id_student', 'code_module'])[col].transform(
                lambda x: x.expanding().mean()
            )
            df[f'{col}_deviation_from_avg'] = df[col] - personal_avg
        
        # Learning burst indicator (whether activity this week is significantly higher than average)
        rolling_mean = df.groupby(['id_student', 'code_module'])['total_weekly_clicks'].transform(
            lambda x: x.rolling(4, min_periods=1).mean()
        )
        df['learning_burst_indicator'] = (df['total_weekly_clicks'] > rolling_mean * 2).astype(int)
        
        # Engagement drop warning
        rolling_mean_3 = df.groupby(['id_student', 'code_module'])['total_weekly_clicks'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df['engagement_drop_flag'] = (df['total_weekly_clicks'] < rolling_mean_3 * 0.5).astype(int)
        
        return df
    
    # === 5. Historical performance features ===(Aborted due to performance issues)
    def add_historical_performance_features(df, raw):
        # Merge historical assessment information
        assessments = raw['assessments']
        student_assessments = raw['studentAssessment']
        
        # Calculate historical performance for each student in each course
        hist_performance = (
            student_assessments
            .merge(assessments[['id_assessment', 'code_module', 'code_presentation', 'date']], 
                   on='id_assessment', how='left')
            .sort_values(['id_student', 'code_module', 'code_presentation', 'date'])
        )
        
        # Calculate assessment statistics up to current week
        hist_performance['assess_week'] = hist_performance['date'] // 7
        
        # Calculate historical performance for each student-course-week combination
        perf_stats = []
        for _, row in df.iterrows():
            student_id = row['id_student']
            module = row['code_module']
            presentation = row['code_presentation']
            current_week = row['week']
            
            # Get assessments up to current week
            past_assessments = hist_performance[
                (hist_performance['id_student'] == student_id) &
                (hist_performance['code_module'] == module) &
                (hist_performance['code_presentation'] == presentation) &
                (hist_performance['assess_week'] <= current_week)
            ]
            
            if len(past_assessments) > 0:
                # Filter out NaN scores for calculations
                valid_scores = past_assessments.dropna(subset=['score'])
                
                if len(valid_scores) > 0:
                    avg_score = valid_scores['score'].mean()
                    
                    # Calculate trend only if we have at least 2 valid scores with different weeks
                    score_trend = 0
                    if len(valid_scores) >= 2:
                        scores = valid_scores['score'].values
                        weeks = valid_scores['assess_week'].values
                        
                        # Check if there's variation in both scores and weeks
                        if len(np.unique(scores)) > 1 and len(np.unique(weeks)) > 1:
                            try:
                                # Use simple linear regression slope instead of correlation
                                # to avoid division by zero issues
                                x_mean = np.mean(weeks)
                                y_mean = np.mean(scores)
                                numerator = np.sum((weeks - x_mean) * (scores - y_mean))
                                denominator = np.sum((weeks - x_mean) ** 2)
                                
                                if denominator != 0:
                                    score_trend = numerator / denominator
                                else:
                                    score_trend = 0
                            except:
                                score_trend = 0
                        else:
                            score_trend = 0
                else:
                    avg_score = np.nan
                    score_trend = 0
                
                # Count missed assessments (NaN scores)
                missed_count = past_assessments['score'].isna().sum()
            else:
                avg_score = np.nan
                score_trend = 0
                missed_count = 0
            
            perf_stats.append({
                'avg_score_to_date': avg_score,
                'score_trend': score_trend,
                'missed_assessments_count': missed_count
            })
        
        perf_df = pd.DataFrame(perf_stats)
        df = pd.concat([df, perf_df], axis=1)
        
        return df
    
    # === 6. Demographic interaction features ===
    def add_demographic_interactions(df):
        # Age-education interaction
        df['age_education_interaction'] = df['age_band'].astype(str) + '_' + df['highest_education'].astype(str)
        
        # Region-deprivation interaction
        df['region_deprivation_interaction'] = df['region'].astype(str) + '_' + df['imd_band'].astype(str)
        
        # Repeat attempt risk assessment
        df['repeat_risk_score'] = df['num_of_prev_attempts'].fillna(0) * (df['studied_credits'] / 60)  # Normalized to 60 credits
        
        # Study load intensity assessment
        df['study_load_intensity'] = df['studied_credits'] / 60  # Relative to standard 60 credits load
        
        return df
    
    
    print("Adding cross-course features...")
    df = add_cross_course_features(df)
    
    print("Adding temporal features...")
    df = add_temporal_features(df, raw)

    
    print("Adding demographic interactions...")
    df = add_demographic_interactions(df)
    
    # Clean up auxiliary columns
    df = df.drop(columns=['module_presentation_length'], errors='ignore')
    
    print(f"Feature engineering completed. Shape: {df.shape}")
    return df









def read_csv(filename: str, data_dir: str) -> pd.DataFrame:
    """
    Unified function to read a CSV file and return a DataFrame.
    filename: Name of the file including extension, e.g., 'assessments.csv'.
    data_dir: Directory path, e.g., 'data/raw/anonymisedData'.

    """
    path = os.path.join(data_dir, filename)
    return pd.read_csv(path)


def load_all_raw(data_dir: str) -> dict:
    """
    Reads all original CSV tables in the specified data_dir and returns them as a dictionary.
    Each key is the filename with the '.csv' extension removed.
    """

    tables = {}
    for fn in FILENAMES:
        key = fn.replace('.csv', '')
        tables[key] = read_csv(fn, data_dir)
    return tables


def build_weekly_dataset(raw: dict) -> pd.DataFrame:
    """
    Constructs a weekly-aggregated DataFrame (train_weekly) from multiple raw tables.
    Includes click-based features, demographic features, and a target label (target_fail) indicating failure in the next assessment.
    """
    assessments = raw['assessments']
    courses     = raw['courses']
    studentAssessment = raw['studentAssessment']
    studentVle  = raw['studentVle']
    vle         = raw['vle']
    studentInfo = raw['studentInfo']

        
    # === Step 1: prepare assess_df ===
    # merge to get assessment date and score, ascending order
    assess_df = (
        studentAssessment
        .merge(
            assessments[["id_assessment", "code_module", "code_presentation", "date"]],
            on="id_assessment",
            how="left"
        )
        .rename(columns={"date": "assess_date"})
        [["id_student", "code_module", "code_presentation", "assess_date", "score"]]
        .sort_values(
            ["id_student", "code_module", "code_presentation", "assess_date"]
        )
        .reset_index(drop=True)
    )

    # === Step 2: prepare vle weekly sum_click feature（vle_features） ===
    # 2.1 merge studentVle and vle to add activity_type
    sv = studentVle.merge(
        vle[["id_site", "activity_type"]],
        on="id_site",
        how="left"
    )

    # 2.2 calculate week num
    sv["week"] = sv["date"] // 7

    # 2.3 calculate sum_click according to week and activity_type 
    weekly = (
        sv
        .groupby(
            ["id_student", "code_module", "code_presentation", "activity_type", "week"],
            as_index=False
        )["sum_click"]
        .sum()
    )

    # 2.4 Sort and apply cumulative sum to get cumulative clicks up to the current week
    weekly = weekly.sort_values(
        ["id_student", "code_module", "code_presentation", "activity_type", "week"]
    )
    weekly["cum_click"] = (
        weekly
        .groupby(
            ["id_student", "code_module", "code_presentation", "activity_type"]
        )["sum_click"]
        .cumsum()
    )

    # 2.5 Pivot both sum_click (clicks this week) and cum_click (cumulative clicks up to this week)
    vle_features = (
        weekly
        .pivot_table(
            index=["id_student", "code_module", "code_presentation", "week"],
            columns="activity_type",
            values=["sum_click", "cum_click"],
            fill_value=0
        )
        .reset_index()
    )

    # Flatten MultiIndex column names:
    # Add prefix only when 'act' (i.e., activity_type) is not empty; otherwise, keep the original column name
    vle_features.columns = [
        f"{val_type}_{act}" if act else val_type
        for val_type, act in vle_features.columns
    ]


    # ——— Before forcing conversion to int, fill missing assess_date ———
    # 1) Merge the length of each module-presentation
    lengths = courses[["code_module", "code_presentation", "module_presentation_length"]]
    assess_df = assess_df.merge(lengths, on=["code_module","code_presentation"], how="left")

    # 2) Use the module length to fill all missing assess_date values
    assess_df["assess_date"] = assess_df["assess_date"].fillna(assess_df["module_presentation_length"])

    # 3) Drop the auxiliary 'length' column
    assess_df = assess_df.drop(columns="module_presentation_length")

    # After Step 1 prep of the assessment table, force assess_date to integer type
    assess_df["assess_date"] = assess_df["assess_date"].astype(int)


    # === Step 3: Identify the "next assessment" in one step and mark the target ===
    # 3.1 Prepare the left table for merge_asof
    vle_df = vle_features.copy()

    vle_df["week_end"] = vle_df["week"] * 7

    # 3.2 Sort the table to ensure correct matching in merge_asof
    vle_df = vle_df.sort_values(
        ["id_student", "code_module", "code_presentation", "week_end"]
    )

    # 1) Globally sort by week_end in ascending order
    vle_df = vle_df.sort_values("week_end")

    # 2) Globally sort by assess_date in ascending order (and ensure it's of type int64)
    assess_df = assess_df.sort_values("assess_date")
    assess_df["assess_date"] = assess_df["assess_date"].astype("int64")


    # Now merge_asof can be applied correctly
    merged = pd.merge_asof(
        vle_df,
        assess_df,
        by=["id_student", "code_module", "code_presentation"],
        left_on="week_end",
        right_on="assess_date",
        direction="forward",
        suffixes=("", "_next")
    )


    # 1) First calculate which week each assessment belongs to, then take the maximum week
    last_week = (
        assess_df
        .assign(assess_week=lambda df: df["assess_date"] // 7)
        .groupby(
            ["id_student", "code_module", "code_presentation"],
            as_index=False
        )["assess_week"]
        .max()
        .rename(columns={"assess_week": "last_assess_week"})
    )

    # 2) Merge last_assess_week into the merged dataframe
    merged = merged.merge(
        last_week,
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )

    # 3) Filter: keep records either before the last assessment week or those with no assessments at all
    merged = merged[
        (merged["week"] <= merged["last_assess_week"]) 
        | merged["last_assess_week"].isna()
    ].copy()


    merged = merged.drop(columns="last_assess_week")


    # 3.4 Generate target: mark 1 if the next assessment is missing or has a score < 40, otherwise mark 0
    merged["target_fail"] = (
    merged["score"].isna()    
    | (merged["score"] < 40) 
    ).astype(int)


    # === Step 4: Merge demographic features ===
    demo = (
        studentInfo[[
            "id_student", "code_module", "code_presentation",
            "gender", "region", "highest_education", "imd_band",
            "age_band", "num_of_prev_attempts", "studied_credits",
            "disability"
        ]]
        .drop_duplicates()
    )

    train_weekly = (
        merged
        .merge(
            demo,
            on=["id_student", "code_module", "code_presentation"],
            how="left"
        )
        .assign(week_start_day=lambda df: df["week"] * 7)
        # also ignore redundant cols such as assess_date、score、score_next、week_end
    )

    train_weekly = train_weekly.drop(columns=[
    "assess_date",
    "score"])

    train_weekly["imd_band"] = train_weekly["imd_band"].fillna("Unknown")

    # Add feature engineering before return
    print("Starting feature engineering...")
    train_weekly = add_engineered_features(train_weekly, raw)

    return train_weekly








def load_weekly(data_dir: str = 'data/raw/anonymisedData') -> pd.DataFrame:
    """
    Main interface: given the directory of raw CSV files, return the cleaned and merged train_weekly DataFrame
    """
    raw = load_all_raw(data_dir)
    weekly = build_weekly_dataset(raw)
    return weekly


if __name__ == '__main__':
    df = load_weekly()
    print('train_weekly shape:', df.shape)
    print(df.head())
    print('Missing values per column:')
    print(df.isnull().sum())

    # For debugging: temporarily save df to CSV for inspection; comment out when not needed
    df.to_csv('train_weekly.csv', index=False)
    import os
    print("Current working directory:", os.getcwd())
