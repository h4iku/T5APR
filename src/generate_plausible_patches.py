import json
from pathlib import Path

import pandas as pd

from .configs import generated_assets_root, results_dir

# Config
beam_size = 100
assessment = False


benchmarks = {
    "Bears": ("Bears.jsonl", "java"),
    "BugAID": ("BugAID.jsonl", "javascript"),
    "Codeflaws": ("Codeflaws.jsonl", "c"),
    "Defects4J-v1.2": ("Defects4J.jsonl", "java"),
    "Defects4J-v2.0": ("Defects4J.jsonl", "java"),
    "ManyBugs": ("ManyBugs.jsonl", "c"),
    "QuixBugs-Java": ("QuixBugs-Java.jsonl", "java"),
    "QuixBugs-Python": ("QuixBugs-Python.jsonl", "python"),
}

not_hunks_bugids = {
    "Lang 10",
    "Math 65",
    "Math 81",
    "Lang 4",
    "Lang 63",
    "Time 7",
    "Cli 18",
    "Lang 46",
}


def get_plausibles(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if benchmark in ["ManyBugs", "BugAID"]:
        return df.loc[df["correct"]]
    else:
        return df.loc[df["plausible"]]


def get_bugid_patches(df: pd.DataFrame, bugid: str) -> pd.DataFrame:
    """Returns the subset of `df` containing plausible patches for a specific bugid and all its hunks"""
    return df.loc[df["bugid"] == bugid]


def save_patches(df: pd.DataFrame, benchmark: str, patches_path: Path) -> None:
    for bugid, bug_patches in df.groupby("bugid"):
        bug_patches = bug_patches.reset_index(drop=True)
        plausible_patches = get_plausibles(bug_patches, benchmark)
        for bug_patch in plausible_patches.itertuples(name="Patch"):
            # This is the mechanism to check if a patch is multi-hunk or not.
            multi_hunk = bug_patch.hunk == 1 and bugid not in not_hunks_bugids
            patches_dir = (
                results_dir
                / "/".join(patches_path.parts[-3:-1])
                / f"plausible_patches_{beam_size}"
            )
            save_dir = patches_dir / bugid
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(
                save_dir / f"{str(bug_patch.Index).zfill(3)}.json", "w"
            ) as patch_file:
                json.dump(
                    {
                        "exact_match": bug_patch.exact_match,
                        "correct": bug_patch.correct,
                        "plausible": bug_patch.plausible
                        if benchmark not in ["ManyBugs", "BugAID"]
                        else None,
                        "multi-hunk": multi_hunk,
                        "source": bug_patch.source,
                        "patch": bug_patch.decoded_sequences,
                        "target": bug_patch.target,
                    },
                    patch_file,
                    indent=4,
                )


def main():
    for benchmark, meta in benchmarks.items():
        for multi in [True, False]:
            model = "multi" if multi else meta[1]
            patches_path = (
                generated_assets_root
                / benchmark
                / f"outputs-{model}"
                / f"validated_reranked_candidates_{beam_size}.jsonl"
            )
            patches_df = pd.read_json(patches_path, orient="records", lines=True)
            save_patches(patches_df, benchmark, patches_path)


if __name__ == "__main__":
    main()
