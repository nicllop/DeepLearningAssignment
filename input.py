import pandas as pd
from names import *
from utils import *


def load_feature_df_from_csv(file: str):
    load_feature_df = pd.read_csv(file, index_col=0, dtype=str)

    # Format time F1 feature.
    if all(load_feature_df[DATE_F1].apply(is_number)):
        load_feature_df[DATE_F1] = load_feature_df[DATE_F1].apply(
            lambda x: str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:8])

    # Float columns retype as float
    for column in load_feature_df.columns.to_list():
        if all(load_feature_df[column].apply(is_number)):
            load_feature_df = load_feature_df.astype({column: float})

    return load_feature_df