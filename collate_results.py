import glob
import os

import pandas as pd

folder = "combined/"

datasets = ["mnist", "cifar"]
modes = ["fine", "coarse"]
datatypes = ["margins", "test_accuracy", "training_accuracy", "training_loss"]


def get_data(folder, limit_to=None):
    dataframes_dict = {dataset: {mode: {datatype: [] for datatype in datatypes} for mode in modes} for dataset in datasets}

    # Iterate through each directory in the runs directory
    for dir_path in glob.glob(os.path.join(folder, "*")):
        if os.path.isdir(dir_path):
            dir_name = os.path.basename(dir_path)
            dataset, mode = dir_name.split("_")[-3:-1]

            for datatype in datatypes:
                df = pd.read_csv(os.path.join(dir_path, f"{datatype}.csv"), index_col=0)
                if limit_to is None or limit_to > len(dataframes_dict[dataset][mode][datatype]):
                    dataframes_dict[dataset][mode][datatype].append(df)
    return dataframes_dict


dataframes_dict = get_data(folder)

# Check that all lists in dataframes have the same length
lengths = [len(dataframes_dict[dataset][mode][datatype]) for datatype in datatypes for dataset in datasets for mode in modes]
if len(set(lengths)) != 1:
    for dataset in datasets:
        for mode in modes:
            for datatype in datatypes:
                print(f"{dataset} {mode} {datatype}: {len(dataframes_dict[dataset][mode][datatype])}")

# get data but limit to minimum number of runs
dataframes_dict = get_data(folder, limit_to=min(lengths))
lengths = [len(dataframes_dict[dataset][mode][datatype]) for datatype in datatypes for dataset in datasets for mode in modes]
print(f"Found {lengths[0]} runs for each dataset, mode and datatype")


os.makedirs("aggregated_data", exist_ok=True)

for dataset in datasets:
    for mode in modes:
        for datatype in datatypes:
            concat_df = pd.concat(
                dataframes_dict[dataset][mode][datatype], axis=0, keys=range(len(dataframes_dict[dataset][mode][datatype]))
            )

            aggregations = {
                "mean": concat_df.groupby(level=1).mean(),
                "median": concat_df.groupby(level=1).median(),
                "min": concat_df.groupby(level=1).min(),
                "max": concat_df.groupby(level=1).max(),
                "std": concat_df.groupby(level=1).std(),
                "upper_quartile": concat_df.groupby(level=1).quantile(0.75),
                "lower_quartile": concat_df.groupby(level=1).quantile(0.25),
            }

            combined_df = pd.concat([df.add_suffix(f"_{key}") for key, df in aggregations.items()], axis=1)
            combined_df.to_csv(f"aggregated_data/{dataset}_{mode}_{datatype}.csv")
