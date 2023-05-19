import pandas as pd
from names import *

def is_number(string:str):
    try:
        float(string)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

# def is_float(string:str):
#     if DOT in string:
#         return True
#     else:
#         return False
#
#
# def type_value(value):
#     if is_number(value) and is_float(value):
#         return float(value)
#     elif is_number(value) and not is_float(value):
#         return int(value)
#     else:
#         return value

def drop_columns_if_exist(df:pd.DataFrame, columns:list):
    output_df = df.copy(deep=True)
    for column in columns:
        if column in df.columns:
            output_df.drop(column, axis=1, inplace=True)
    return output_df

def generate_defect_report(feature_df:pd.DataFrame):
    if DEFECT_CLS_F3 in feature_df.columns:
        print(feature_df[DEFECT_CLS_F3].value_counts())
    elif DEFECT_OKKO_F3 in feature_df.columns:
        print(feature_df[DEFECT_OKKO_F3].value_counts())
    elif DEFECT_INT_F3 in feature_df.columns:
        print(feature_df[DEFECT_INT_F3].value_counts())
    else:
        print("No defect report")