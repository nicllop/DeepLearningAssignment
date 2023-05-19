import pandas as pd
import numpy as np
from utils import *
from names import *
from input import *


def prep_run(file:str = "machineData.csv", data_aumentation:str=UNDERSAMPLE, show_reports:bool=False):

    input_df = load_feature_df_from_csv(file)

    feature_df = merge_date_feature(input_df)
    feature_df = generate_cyclical_time(feature_df)

    feature_df = object_column_to_categorical(feature_df, "ActStsMach_F1")
    feature_df = object_column_to_categorical(feature_df, "PartNumber_F2")
    feature_df = object_column_to_categorical(feature_df, "Mold_F2")

    print("Columns pre redundant filter:", feature_df.shape)

    feature_df = filter_redundant_features(feature_df)

    print("Columns post redundant filter:", feature_df.shape)

    ## PREPROCESS DATA

    #Comentar esto si no es necesario
    # feature_df = diff_by_each_cavity(feature_df, lag=2, features=[HOUR_COS, HOUR_SIN])
    # feature_df = diff_by_each_cavity(feature_df, lag=4, features=[HOUR_COS, HOUR_SIN])
    # feature_df = diff_by_each_cavity(feature_df, lag=8, features=[HOUR_COS, HOUR_SIN])
    # feature_df = diff_by_each_cavity(feature_df, lag=2, regex="ActStsMach_F1")
    # feature_df = diff_by_each_cavity(feature_df, lag=4, regex="ActStsMach_F1")
    # feature_df = diff_by_each_cavity(feature_df, lag=8, regex="ActStsMach_F1")

    target_column = DEFECT_INT_F3
    no_target_column = [DEFECT_CLS_F3, DEFECT_OKKO_F3]

    if data_aumentation == OVERSAMPLE:
        aumentated_df = oversample_df(feature_df, target_column, ko_oversample=15)

        print("------------------------------------") if show_reports else None
        print("-------- OVERSAMPLED REPORT --------") if show_reports else None
        print("------------------------------------") if show_reports else None
        generate_defect_report(aumentated_df) if show_reports else None
        total_kos = aumentated_df[DEFECT_OKKO_F3].value_counts()[KO] if show_reports else None
        print("------------------------------------") if show_reports else None
        print("-------- TOTAL KO: " + str(total_kos) + " --------") if show_reports else None
        print("------------------------------------" + "\n\n") if show_reports else None

    elif data_aumentation == UNDERSAMPLE:
        aumentated_df = undersample_df(feature_df, target_column, ok_undersample=0.1)
        print("-------------------------------------") if show_reports else None
        print("-------- UNDERSAMPLED REPORT --------") if show_reports else None
        print("-------------------------------------") if show_reports else None
        generate_defect_report(aumentated_df) if show_reports else None
        total_kos = aumentated_df[DEFECT_OKKO_F3].value_counts()[KO] if show_reports else None
        print("-------------------------------------") if show_reports else None
        print("-------- TOTAL KO: " + str(total_kos) + " --------") if show_reports else None
        print("-------------------------------------" + "\n\n") if show_reports else None

    else:
        aumentated_df = feature_df.copy(deep=True)

    final_df = drop_columns_if_exist(aumentated_df, no_target_column)

    #ORDER BY DATE!!!!!!!!!!!!!!!!!!!!!

    final_df.drop(DATE_AND_TIME_F1, axis=1, inplace=True)
    # final_df = drop_columns_if_exist(df_temp, DATE_AND_TIME_F1)

    pd.set_option('display.max_columns', None)
    print("Dataset por process: ")
    print(final_df)
    return final_df




def merge_date_feature(feature_df: pd.DataFrame):
    df = feature_df.copy(deep=True)
    df = drop_columns_if_exist(df, [DATE_F2, DATE_AND_TIME_F1])  #Mirar para quitar date F2

    date_and_time_f1_value = df[DATE_F1].map(str) + " " + df[TIME_F1].map(str)
    df.insert(loc=2, column=DATE_AND_TIME_F1, value=date_and_time_f1_value)
    df[DATE_AND_TIME_F1] = pd.to_datetime(df[DATE_AND_TIME_F1], format="%Y-%m-%d %H:%M:%S")

    df = drop_columns_if_exist(df, [TIME_F1, DATE_F1])
    return df.sort_values(by=[DATE_AND_TIME_F1])


def generate_cyclical_time(feature_df: pd.DataFrame):
    df = feature_df.copy(deep=True)
    cyclical_columns = [HOUR_SIN, HOUR_COS, DAY_SIN, DAY_COS]
    df = drop_columns_if_exist(df, cyclical_columns)

    seconds_in_day = 24 * 60 * 60
    hour_to_rad = lambda dt: 2 * np.pi * (dt.hour * 3600 + dt.minute * 60 + dt.second) / seconds_in_day
    hour_sin_values = df[DATE_AND_TIME_F1].map(lambda x: np.sin(hour_to_rad(x)))
    hour_cos_values = df[DATE_AND_TIME_F1].map(lambda x: np.cos(hour_to_rad(x)))

    seconds_in_month = 30 * 24 * 60 * 60
    day_to_rad = lambda dt: 2 * np.pi * (dt.day * seconds_in_day + dt.hour * 3600 + dt.minute * 60 + dt.second) / seconds_in_month
    day_to_rad_limit = lambda dt: min(day_to_rad(dt), 2 * np.pi)
    day_sin_values = df[DATE_AND_TIME_F1].map(lambda x: np.sin(day_to_rad_limit(x)))
    day_cos_values = df[DATE_AND_TIME_F1].map(lambda x: np.cos(day_to_rad_limit(x)))

    df.insert(loc=3, column=HOUR_SIN, value=hour_sin_values)
    df.insert(loc=4, column=HOUR_COS, value=hour_cos_values)
    df.insert(loc=5, column=DAY_SIN, value=day_sin_values)
    df.insert(loc=6, column=DAY_COS, value=day_cos_values)

    #df = drop_columns_if_exist(df, [DATE_AND_TIME_F1])
    return df


def object_column_to_categorical(feature_df: pd.DataFrame, column: str, set_of_values: list = None):
    df = feature_df.copy(deep=True)
    df[column] = df[column].astype(str)

    if set_of_values is None:
        set_of_values = set(df[column].values)

    value_columns = [column + "_is_" + str(value) for value in set_of_values]
    df = drop_columns_if_exist(df, value_columns)

    loc_column = df.columns.get_loc(column)

    for value in set_of_values:
        loc_column += 1
        categorical_values = (df[column] == value) * 1.0
        df.insert(loc=loc_column, column=column + "_is_" + str(value), value=categorical_values)

    df = drop_columns_if_exist(df, [column])
    return df


def diff_all_features(feature_df, lag: int = 1, features: list = None, regex: str = None):
    new_frame = feature_df.copy(deep=True)

    if features is None and regex is not None:
        features = list(filter(lambda name: regex in name, new_frame.columns))
    elif features is not None and regex is None:
        features = list(filter(lambda name: name in features, new_frame.columns))
    else:
        features = list(filter(lambda name: regex in name or name in features, new_frame.columns))

    no_diff_features = [DEFECT_CLS_F3, DEFECT_OKKO_F3, CAVITY_F2, DEFECT_INT_F3]
    features = list(filter(lambda name: "_diff_" not in name and name not in no_diff_features, features))
    diff_frame = new_frame[features].diff(periods=lag)

    # Add _diff_ + lag prefix to diff_frame columns
    diff_frame.columns = [column + "_diff_" + str(lag) for column in diff_frame.columns]
    return pd.concat([feature_df, diff_frame], axis=1).dropna()


def diff_by_each_cavity(feature_df, lag: int = 1, features: list = None, regex: str = None):
    df = feature_df.copy(deep=True)
    df = df.groupby(CAVITY_F2, as_index=False).apply(lambda x: diff_all_features(x, lag, features, regex))
    df.set_index(np.array(list(map(lambda x: x[1], df.index))), inplace=True)
    return df


def oversample_df(feature_df:pd.DataFrame, target_column:str, ko_oversample:float=20.0):
    assert ko_oversample >= 1.0, "ko_oversample must be greater than 1.0"

    original_df = feature_df.copy(deep=True)
    ko_df = original_df[original_df[target_column] == 0]
    ok_df = original_df[original_df[target_column] == 1]
    ko_df_oversampled = ko_df.sample(frac=ko_oversample, replace=True).reset_index(drop=True)
    df_oversampled = pd.concat([ko_df_oversampled, ok_df], axis=0).sample(frac=1).reset_index(drop=True)
    return df_oversampled



def undersample_df(feature_df:pd.DataFrame, target_column:str, ok_undersample:float=0.1):
    assert ok_undersample <= 1.0, "ok_undersample must be less than 1.0"

    original_df = feature_df.copy(deep=True)
    ko_df = original_df[original_df[target_column] == 0]
    ok_df = original_df[original_df[target_column] == 1]
    ok_df_undersampled = ok_df.sample(frac=ok_undersample).reset_index(drop=True)
    df_undersampled = pd.concat([ok_df_undersampled, ko_df], axis=0).sample(frac=1).reset_index(drop=True)
    return df_undersampled


def filter_redundant_features(input_df):
    df1 = input_df.copy(deep=True)

    apart = df1.select_dtypes(include='object')
    df = df1.drop(df1.select_dtypes(include='object'), axis=1)

    corr_df = df.corr()

    high_corr_features = []
    feature_already_checked = []
    for index in corr_df.index:
        for column in corr_df.columns:
            value = corr_df.loc[index, column]
            if value == 1 and index != column and column not in feature_already_checked:
                high_corr_features.append((index, column))
        feature_already_checked.append(index)

    redundant_features = []
    for features in high_corr_features:
        if (df[features[0]] == df[features[1]]).min():
            redundant_features.append(features[0])

    redundant_features = list(set(redundant_features))
    df.drop(redundant_features, axis=1, inplace=True)

    joined_df = pd.concat([df, apart], axis=1)

    return joined_df


