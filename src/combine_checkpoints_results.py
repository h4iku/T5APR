from itertools import chain

import numpy as np
import pandas as pd
from bugaid_datasets_conf import bugaid_gen_dir, codeflaws_gen_dir, manybugs_gen_dir
from d4j_datasets_conf import bears_gen_dir, d4j_gen_dir
from datasets_conf import quixbugs_genjava_dir, quixbugs_genpy_dir

# Config
dataset = "QuixBugs-Python"
multi = True

if dataset == "QuixBugs-Python":
    gen_dir = quixbugs_genpy_dir
    model = "multi" if multi else "python"
    bugs_metadata_file = "QuixBugs_Python.jsonl"
elif dataset == "QuixBugs-Java":
    gen_dir = quixbugs_genjava_dir
    model = "multi" if multi else "java"
    bugs_metadata_file = "QuixBugs_Java.jsonl"
elif dataset == "Defects4J":
    gen_dir = d4j_gen_dir
    model = "multi" if multi else "java"
    bugs_metadata_file = "Defects4J.jsonl"
elif dataset == "BugAID":
    gen_dir = bugaid_gen_dir
    model = "multi" if multi else "javascript"
    bugs_metadata_file = "BugAid.jsonl"
elif dataset == "Codeflaws":
    gen_dir = codeflaws_gen_dir
    model = "multi" if multi else "c"
    bugs_metadata_file = "Codeflaws.jsonl"
elif dataset == "ManyBugs":
    gen_dir = manybugs_gen_dir
    model = "multi" if multi else "c"
    bugs_metadata_file = "ManyBugs.jsonl"
elif dataset == "Bears":
    gen_dir = bears_gen_dir
    model = "multi" if multi else "java"
    bugs_metadata_file = "Bears.jsonl"
else:
    print("Wrong dataset name")

output_dir = gen_dir / f"outputs-{model}"

output_size = 100
num_checkpoints = 5

rem_file_path = gen_dir / "rem.txt"
add_file_path = gen_dir / "add.txt"

with (
    open(rem_file_path) as rem_file,
    open(add_file_path) as add_file,
):
    sources = [src.strip() for src in rem_file]
    targets = [tgt.strip() for tgt in add_file]


def add_source_target(df: pd.DataFrame) -> pd.DataFrame:
    rem_file_path = gen_dir / "rem.txt"
    add_file_path = gen_dir / "add.txt"

    with (
        open(rem_file_path) as rem_file,
        open(add_file_path) as add_file,
    ):
        sources = [src.strip() for src in rem_file]
        targets = [tgt.strip() for tgt in add_file]

    checkpoints_num = len(df.value_counts("checkpoint"))
    assert checkpoints_num == num_checkpoints

    df["source"] = list(chain(*[[s] * output_size for s in sources])) * num_checkpoints
    df["target"] = list(chain(*[[t] * output_size for t in targets])) * num_checkpoints

    for bugid, group in df.groupby("bugid"):
        assert len(group["target"].unique()) <= len(group["hunk"].unique())

    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df["decoded_sequences"] = df["decoded_sequences"].str.strip()
    df["normalized_patch"] = df["decoded_sequences"].str.split().str.join(sep=" ")
    df["normalized_source"] = df["source"].str.split().str.join(sep=" ")
    df["normalized_target"] = df["target"].str.split().str.join(sep=" ")
    return df


def create_empty_patch(patch_sample: pd.Series) -> pd.DataFrame:
    patch_sample.loc["decoded_sequences"] = ""
    patch_sample.loc["sequences_scores"] = -np.inf
    patch_sample.loc["normalized_patch"] = ""
    patch_sample.loc["checkpoint"] = "manual"

    return pd.DataFrame([patch_sample])


# TODO: Better name -> combine_candidates
def deduplicate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """deduplicate, sort and combine candidate patches of different checkpoints"""

    src_neq_df = df.loc[df["normalized_patch"] != df["normalized_source"]]

    dfs = []
    for _, subset_df in src_neq_df.groupby(["bugid", "hunk", "checkpoint"]):
        subset_df["rank"] = subset_df.reset_index(drop=True).index
        dfs.append(subset_df)

    ranked_df = pd.concat(dfs)

    # Should sort based on scores before deduplication for `keep=first` to take effect
    sorted_df = ranked_df.sort_values(
        by=["bugid", "hunk", "rank", "sequences_scores"],
        ascending=[True, True, True, False],
        inplace=False,
        ignore_index=True,
    )
    deduped_df = sorted_df.drop_duplicates(
        subset=["bugid", "hunk", "normalized_patch"],
        inplace=False,
        ignore_index=True,
    )

    # Adding empty patch to hunks
    concat_dfs = []
    grouped_df = deduped_df.groupby(["bugid", "hunk"])
    for _, group_df in grouped_df:
        if (
            "" not in group_df["normalized_patch"].values
            and group_df["normalized_source"].values[0]
        ):
            empty_patch = create_empty_patch(group_df.iloc[-1].copy())
            concat_dfs.append(pd.concat([group_df, empty_patch], ignore_index=True))
        else:
            concat_dfs.append(group_df)

    return pd.concat(concat_dfs, ignore_index=True)


def set_exact_matches(df: pd.DataFrame) -> pd.DataFrame:
    exact_match_condition = df["normalized_patch"] == df["normalized_target"]
    df["exact_match"] = exact_match_condition
    df["correct"] = exact_match_condition
    return df


def main():
    checkpoints_results = pd.read_json(
        output_dir / f"sequences_{output_size}.jsonl",
        orient="records",
        lines=True,
    )

    assert len(checkpoints_results.value_counts("checkpoint")) == num_checkpoints

    column_index = (
        checkpoints_results.columns[-2:].to_list()
        + checkpoints_results.columns[:-2].to_list()
    )

    checkpoints_results = checkpoints_results[column_index]
    print("All:", len(checkpoints_results))
    add_source_target(checkpoints_results)

    deduped_df = deduplicate_candidates(normalize(checkpoints_results))
    print("Deduped:", len(deduped_df))
    set_exact_matches(deduped_df)

    deduped_df.to_json(
        output_dir / f"final_candidates_{output_size}.jsonl",
        orient="records",
        lines=True,
    )


if __name__ == "__main__":
    main()
