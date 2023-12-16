import json
from itertools import chain

import numpy as np
import pandas as pd
from bugaid_datasets_conf import bugaid_gen_dir, codeflaws_gen_dir, manybugs_gen_dir
from d4j_datasets_conf import bears_gen_dir, d4j_gen_dir
from datasets_conf import quixbugs_genjava_dir, quixbugs_genpy_dir
from tqdm import tqdm

# Config
dataset = "QuixBugs-Python"
multi = True
rerank_method = 2
empty_last = False

if dataset == "QuixBugs-Python":
    gen_dir = quixbugs_genpy_dir
    model = "multi-full" if multi else "python"
    bugs_metadata_file = "QuixBugs_Python.jsonl"
elif dataset == "QuixBugs-Java":
    gen_dir = quixbugs_genjava_dir
    model = "multi-full" if multi else "java"
    bugs_metadata_file = "QuixBugs_Java.jsonl"
elif dataset == "Defects4J":
    gen_dir = d4j_gen_dir
    model = "multi-full" if multi else "java"
    bugs_metadata_file = "Defects4J.jsonl"
elif dataset == "BugAID":
    gen_dir = bugaid_gen_dir
    model = "multi-full" if multi else "javascript"
    bugs_metadata_file = "BugAid.jsonl"
elif dataset == "Codeflaws":
    gen_dir = codeflaws_gen_dir
    model = "multi-full" if multi else "c"
    bugs_metadata_file = "Codeflaws.jsonl"
elif dataset == "ManyBugs":
    gen_dir = manybugs_gen_dir
    model = "multi-full" if multi else "c"
    bugs_metadata_file = "ManyBugs.jsonl"
elif dataset == "Bears":
    gen_dir = bears_gen_dir
    model = "multi-full" if multi else "java"
    bugs_metadata_file = "Bears.jsonl"
else:
    print("Wrong dataset name")

output_dir = gen_dir / f"outputs-{model}"
output_size = 100
patches_dir = output_dir / f"plausible_patches_{output_size}"


def add_source_target(df: pd.DataFrame) -> pd.DataFrame:
    rem_file_path = gen_dir / "rem.txt"
    add_file_path = gen_dir / "add.txt"

    with (
        open(rem_file_path) as rem_file,
        open(add_file_path) as add_file,
    ):
        sources = [src.strip() for src in rem_file]
        targets = [tgt.strip() for tgt in add_file]

    num_checkpoints = len(df.value_counts("checkpoint"))

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
    patch_sample.loc["sequences_scores"] = -np.inf if empty_last else 0
    patch_sample.loc["normalized_patch"] = ""
    patch_sample.loc["checkpoint"] = "manual"
    patch_sample.loc["rank"] = output_size if empty_last else 0

    return pd.DataFrame([patch_sample])


def rerank(df: pd.DataFrame, method: int = 3) -> pd.DataFrame:
    """deduplicate, sort and combine candidate patches of different checkpoints"""

    # 3 ways to do:
    # 1. sort based on sequences_scores
    # 2. sort based on rank and sequences_scores, then remove source equality
    # 3. first remove source equality, then sort based on rank and sequences_scores

    if method == 1:
        src_neq_df = df.loc[df["normalized_patch"] != df["normalized_source"]]
        df = src_neq_df.copy()

        sorted_df = df.sort_values(
            by=["bugid", "hunk", "sequences_scores"],
            ascending=[True, True, False],
            inplace=False,
            ignore_index=True,
        )

    elif method == 2:
        dfs = []
        for _, subset_df in tqdm(df.groupby(["bugid", "hunk", "checkpoint"])):
            subset_df["rank"] = subset_df.reset_index(drop=True).index
            dfs.append(subset_df)

        df = pd.concat(dfs)

        # Should sort based on scores before deduplication for `keep=first` to take effect
        sorted_df = df.sort_values(
            by=["bugid", "hunk", "rank", "sequences_scores"],
            ascending=[True, True, True, False],
            inplace=False,
            ignore_index=True,
        )

        src_neq_df = sorted_df.loc[
            sorted_df["normalized_patch"] != sorted_df["normalized_source"]
        ]
        sorted_df = src_neq_df.copy()

    else:
        src_neq_df = df.loc[df["normalized_patch"] != df["normalized_source"]]
        df = src_neq_df.copy()

        dfs = []
        for _, subset_df in tqdm(df.groupby(["bugid", "hunk", "checkpoint"])):
            subset_df["rank"] = subset_df.reset_index(drop=True).index
            dfs.append(subset_df)

        df = pd.concat(dfs)

        # Should sort based on scores before deduplication for `keep=first` to take effect
        sorted_df = df.sort_values(
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
    for bug, group_df in grouped_df:
        if (
            "" not in group_df["normalized_patch"].values
            and group_df["normalized_source"].values[0]
        ):
            empty_patch = create_empty_patch(group_df.iloc[-1].copy())
            if empty_last:
                concat_dfs.append(pd.concat([group_df, empty_patch], ignore_index=True))
            else:
                concat_dfs.append(pd.concat([empty_patch, group_df], ignore_index=True))
        else:
            concat_dfs.append(group_df)

    return pd.concat(concat_dfs, ignore_index=True)


def generate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    bug_hunks = []

    bugs_list = [
        "Chart 4",
        "Chart 26",
        "Closure 2",
        "Closure 19",
        "Closure 40",
        "Closure 66",
        "Closure 102",
        "Lang 4",
        "Lang 7",
        "Lang 10",
        "Lang 46",
        "Lang 55",
        "Lang 63",
        "Math 3",
        "Math 25",
        "Math 65",
        "Math 77",
        "Math 81",
        "Math 89",
        "Time 7",
        "Cli 14",
        "Cli 18",
        "Codec 15",
        "Csv 5",
        "Csv 6",
        "Gson 6",
        "Gson 12",
        "Gson 16",
        "JacksonCore 21",
        "JacksonDatabind 24",
        "JacksonDatabind 49",
        "JacksonDatabind 101",
        "Jsoup 59",
        "Jsoup 85",
        "Jsoup 89",
        "JxPath 20",
        "JxPath 22",
        "Bears-8",
        "Bears-122",
    ]

    for bugid, bug_df in df.groupby("bugid"):
        hunks_num = len(bug_df["hunk"].unique())
        if hunks_num == 1 or bugid in bugs_list:
            if bugid in ["Lang 10", "Math 65"]:
                bug_df = bug_df.loc[bug_df["hunk"] == 1]
            elif bugid in ["Math 81", "Lang 4"]:
                bug_df = bug_df.loc[bug_df["hunk"] == 2]
            elif bugid in ["Lang 63", "Time 7", "Cli 18"]:
                bug_df = bug_df.loc[bug_df["hunk"] == hunks_num - 1]
            elif bugid in ["Lang 46"]:
                bug_df = bug_df.loc[bug_df["hunk"] == hunks_num - 2]
            else:
                bug_df = bug_df.loc[bug_df["hunk"] == 0]

            bug_hunks.append(bug_df)

        else:
            merged_df = bug_df.loc[
                bug_df["hunk"] == 0,
                ["sequences_scores", "decoded_sequences", "normalized_patch"],
            ].merge(
                bug_df.loc[
                    bug_df["hunk"] == 1,
                    ["sequences_scores", "decoded_sequences", "normalized_patch"],
                ],
                how="inner",
                on="normalized_patch",
            )
            if hunks_num > 2:
                for i in range(2, hunks_num):
                    merged_df = merged_df.merge(
                        bug_df.loc[
                            bug_df["hunk"] == i,
                            [
                                # "rank",
                                "sequences_scores",
                                "decoded_sequences",
                                "normalized_patch",
                            ],
                        ],
                        on="normalized_patch",
                        suffixes=(f"_x{i}", f"_y{i}"),
                    )

            merged_df["merged_score"] = merged_df.max(axis=1, numeric_only=True)

            cp_df = bug_df.loc[
                (bug_df["hunk"] == 1)
                & (bug_df["normalized_patch"].isin(merged_df["normalized_patch"]))
            ].copy()

            for patch, score in merged_df[["normalized_patch", "merged_score"]].values:
                cp_df.loc[
                    cp_df["normalized_patch"] == patch,
                    "sequences_scores",
                ] = score

            cp_df.sort_values(by="sequences_scores", ascending=False, inplace=True)

            bug_hunks.append(cp_df)

    hunk_results = pd.concat(
        bug_hunks,
        ignore_index=True,
    )

    return hunk_results


def set_exact_matches(df: pd.DataFrame) -> pd.DataFrame:
    exact_match_condition = df["normalized_patch"] == df["normalized_target"]
    df["correct"] = exact_match_condition
    df["exact_match"] = exact_match_condition
    return df


def get_plausibles(df: pd.DataFrame) -> pd.DataFrame:
    if dataset in ["ManyBugs", "BugAID"]:
        return df
    else:
        return df.loc[df["plausible"]]


def get_corrects(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["correct"]]


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    patches_df = pd.read_json(
        output_dir / f"all_candidates_{output_size}.jsonl",
        orient="records",
        lines=True,
    )

    new_columns = [c for c in patches_df.columns if c not in df.columns]

    output = df.merge(
        patches_df[["bugid", "normalized_patch"] + new_columns],
        on=["bugid", "normalized_patch"],
    )
    return output


def exact_match_plausibility_check(patches_df: pd.DataFrame) -> None:
    if dataset in ["BugAID", "ManyBugs"]:
        return

    not_plausible = patches_df[patches_df["plausible"] is False]
    exact_match_not_plausible = not_plausible[not_plausible["exact_match"] is True]

    assert (
        exact_match_not_plausible.empty
    ), "There is an exact match that is not marked as plausible"


def set_corrects(df: pd.DataFrame) -> pd.DataFrame:
    df["correct"] = False
    df["exact_match"] = False

    for bug_dir in patches_dir.iterdir():
        for patch_path in bug_dir.iterdir():
            with open(bug_dir / patch_path) as patch_file:
                patch = json.load(patch_file)

                if patch["correct"]:
                    df.loc[
                        (df["bugid"] == bug_dir.name)
                        # & (df["hunk"] == hunk_num)
                        & (df["normalized_patch"] == " ".join(patch["patch"].split())),
                        ["exact_match", "correct"],
                    ] = [patch["exact_match"], patch["correct"]]
    return df


def set_corrects_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df["correct"] = False
    df["exact_match"] = False

    patches_df = pd.read_json(
        output_dir / f"auto_validated_patches_{output_size}.jsonl",
        orient="records",
        lines=True,
    )

    corrects_df = get_corrects(patches_df)

    for ppatch in corrects_df.itertuples(name="Patch"):
        df.loc[
            (df["bugid"] == ppatch.bugid)
            & (df["normalized_patch"] == ppatch.normalized_patch),
            "correct",
        ] = True

    return df


def main():
    # Load all checkpoints results
    checkpoints_results = pd.read_json(
        output_dir / f"sequences_{output_size}.jsonl",
        orient="records",
        lines=True,
    )
    column_index = (
        checkpoints_results.columns[-2:].to_list()
        + checkpoints_results.columns[:-2].to_list()
    )
    checkpoints_results = checkpoints_results[column_index]

    assert len(checkpoints_results.value_counts("checkpoint")) == 5

    add_source_target(checkpoints_results)

    print("All:  ", len(checkpoints_results))
    reranked_df = rerank(normalize(checkpoints_results), method=rerank_method)
    print("Dedup:", len(reranked_df))
    reranked_df = generate_candidates(reranked_df)

    # You can do this both using generated directories (to also check correct type)
    # or using plausible DataFrame (can only check plausible and exact match)
    # or use the primary candidates (can only use exact match)

    # set_exact_matches(reranked_df)
    set_corrects(reranked_df)

    # set_corrects_from_df(reranked_df)
    reranked_df = add_metadata(reranked_df)

    first_em = reranked_df.groupby("bugid").first().value_counts("correct")
    print("1st Exact Match:", first_em.get(True, "0"))

    plausibles_df = get_plausibles(reranked_df)
    exact_match_plausibility_check(reranked_df)
    first_plausible = plausibles_df.groupby("bugid").first().value_counts("correct")
    print("1st Plausible:", first_plausible.get(True, "0"))

    reranked_df.to_json(
        output_dir / f"reranked_candidates_{output_size}.jsonl",
        orient="records",
        lines=True,
    )


if __name__ == "__main__":
    main()
