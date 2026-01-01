import os
import pandas as pd


def load_processed_tables():
    processed_dir = os.path.join('data', 'processed')
    cat_path = os.path.join(processed_dir, 'german_credit_categorical.csv')
    num_path = os.path.join(processed_dir, 'german_credit_numeric.csv')
    if not os.path.exists(cat_path) or not os.path.exists(num_path):
        raise FileNotFoundError(
            'Expected processed files at data/processed/*.csv. '
            'Run process_data.py first.'
        )
    df_cat = pd.read_csv(cat_path)
    df_num = pd.read_csv(num_path)
    if len(df_cat) != len(df_num):
        raise ValueError(
            f"Row count mismatch: cat={len(df_cat)}, num={len(df_num)}"
        )
    return df_cat, df_num


def get_mappings():
    # Mappings based on data/raw/description.txt
    return {
        'checking_account_status': {
            'A11': '< 0 DM',
            'A12': '0 <= ... < 200 DM',
            'A13': '>= 200 DM / salary assign.',
            'A14': 'no checking account',
        },
        'credit_history': {
            'A30': 'no credits/all paid duly',
            'A31': 'all credits here paid duly',
            'A32': 'existing credits paid duly',
            'A33': 'delay in paying off in past',
            'A34': 'critical/other credits exist',
        },
        'purpose': {
            'A40': 'car (new)',
            'A41': 'car (used)',
            'A42': 'furniture/equipment',
            'A43': 'radio/television',
            'A44': 'domestic appliances',
            'A45': 'repairs',
            'A46': 'education',
            # A47 vacaction (does not exist)
            'A48': 'retraining',
            'A49': 'business',
            'A410': 'others',
        },
        'savings_account_bonds': {
            'A61': '< 100 DM',
            'A62': '100 <= ... < 500 DM',
            'A63': '500 <= ... < 1000 DM',
            'A64': '>= 1000 DM',
            'A65': 'unknown/no savings account',
        },
        'present_employment_since': {
            'A71': 'unemployed',
            'A72': '< 1 year',
            'A73': '1 <= ... < 4 years',
            'A74': '4 <= ... < 7 years',
            'A75': '>= 7 years',
        },
        'personal_status_sex': {
            'A91': 'male: divorced/separated',
            'A92': 'female: div/sep/married',
            'A93': 'male: single',
            'A94': 'male: married/widowed',
            'A95': 'female: single',
        },
        'other_debtors_guarantors': {
            'A101': 'none',
            'A102': 'co-applicant',
            'A103': 'guarantor',
        },
        'property': {
            'A121': 'real estate',
            'A122': 'bldg society/life ins.',
            'A123': 'car or other',
            'A124': 'unknown/no property',
        },
        'other_installment_plans': {
            'A141': 'bank',
            'A142': 'stores',
            'A143': 'none',
        },
        'housing': {
            'A151': 'rent',
            'A152': 'own',
            'A153': 'for free',
        },
        'job': {
            'A171': 'unemployed/unskilled non-res.',
            'A172': 'unskilled resident',
            'A173': 'skilled employee/official',
            'A174': 'management/self-employed/etc',
        },
        'telephone': {
            'A191': 'none',
            'A192': 'yes, registered',
        },
        'foreign_worker': {
            'A201': 'yes',
            'A202': 'no',
        },
        'class': {
            1: 'good',
            2: 'bad',
            '1': 'good',
            '2': 'bad',
        },
    }


def merge_and_map():
    df_cat, df_num = load_processed_tables()

    # We'll keep the numeric 'class' (1=good, 2=bad). Drop duplicate numeric
    # columns from df_num that already exist in df_cat to avoid suffix clutter.
    drop_from_num = [
        'duration_months',
        'credit_amount',
        'installment_rate_percentage',
        'present_residence_since',
        'age_years',
        'existing_credits_count',
        'people_liable_for_maintenance',
    ]
    df_num = df_num.drop(
        columns=[c for c in drop_from_num if c in df_num.columns]
    )

    # Also drop 'class' from categorical (we'll rely on numeric one)
    if 'class' in df_cat.columns:
        df_cat = df_cat.drop(columns=['class'])

    # Merge on row index (same order as raw files)
    df = df_cat.join(df_num, how='inner', lsuffix='', rsuffix='_num')

    mappings = get_mappings()
    categorical_cols = [
        'checking_account_status',
        'credit_history',
        'purpose',
        'savings_account_bonds',
        'present_employment_since',
        'personal_status_sex',
        'other_debtors_guarantors',
        'property',
        'other_installment_plans',
        'housing',
        'job',
        'telephone',
        'foreign_worker',
    ]

    # For each categorical column: preserve original codes in *_code, create
    # human-readable column with the original name.
    for col in categorical_cols:
        if col in df.columns:
            code_col = f"{col}_code"
            if code_col in df.columns:
                # Rare collision safety: if *_code already exists, skip rename
                pass
            else:
                df.rename(columns={col: code_col}, inplace=True)
            mapped = df[code_col].map(mappings[col])
            df[col] = mapped.where(mapped.notna(), df[code_col])

    # Map class label from numeric 'class' to a readable 'class_label'
    if 'class' in df.columns:
        df['class_label'] = (
            df['class'].map(mappings['class']).fillna(df['class'])
        )

    out_path = os.path.join(
        'data', 'processed', 'german_credit_final.csv'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(
        (
            f"Merged shape: {df.shape}. "
            f"Saved mapped dataset to {out_path}"
        )
    )


if __name__ == '__main__':
    merge_and_map()
