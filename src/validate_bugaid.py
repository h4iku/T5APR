import contextlib
import json
import os
import shutil
import threading
from collections import ChainMap
from copy import deepcopy
from pathlib import Path
from typing import Optional
from enum import Enum, auto
import timeit
import numpy as np

import joblib
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tree_sitter import Language, Parser

from bugaid_datasets_conf import bugaid_data_dir, bugaid_gen_dir
from datasets_conf import tree_sitter_lib

gen_dir = bugaid_gen_dir
bugs_metadata_file = "BugAid.jsonl"
# output_dir = gen_dir / "outputs-multi-full"
output_dir = gen_dir / "outputs-javascript"
temp_dir = output_dir / "temp"
save_state_dir = output_dir / "save-state"
output_size = 100

rem_file_path = gen_dir / "rem.txt"
add_file_path = gen_dir / "add.txt"

with (
    open(rem_file_path) as rem_file,
    open(add_file_path) as add_file,
):
    sources = [src.strip() for src in rem_file]
    targets = [tgt.strip() for tgt in add_file]


Language.build_library(
    str(tree_sitter_lib / "build/my-languages.so"),
    [
        str(tree_sitter_lib / "vendor/tree-sitter-python"),
        str(tree_sitter_lib / "vendor/tree-sitter-java"),
        str(tree_sitter_lib / "vendor/tree-sitter-javascript"),
        str(tree_sitter_lib / "vendor/tree-sitter-c"),
    ],
)

JAVASCRIPT_LANGUAGE = Language(
    str(tree_sitter_lib / "build/my-languages.so"), "javascript"
)
parser = Parser()
parser.set_language(JAVASCRIPT_LANGUAGE)
query = JAVASCRIPT_LANGUAGE.query(
    """
    (ERROR) @error
    """
)


@contextlib.contextmanager
def change_directory(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


@contextlib.contextmanager
def working_environment(bugid: str, filename: str):
    # Split the bugid to get file names
    project_dir = bugaid_data_dir / bugid / "buggy"
    buggy_file_name = filename

    # Copy files to a working directory
    project_copy_dir = temp_dir / bugid
    copy_dataset_files(project_dir, project_copy_dir)

    target_file_path = project_copy_dir / buggy_file_name

    # Copy initial file to a temp directory
    source_file_path = temp_dir / "sources" / bugid / buggy_file_name
    source_file_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(target_file_path, source_file_path)

    try:
        yield project_copy_dir, source_file_path, target_file_path
    finally:
        shutil.rmtree(project_copy_dir)
        shutil.rmtree(source_file_path.parent)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


def get_bug_hunk_candidates(df: pd.DataFrame, bugid: str, hunk: int) -> pd.DataFrame:
    """Returns the subset of `df` containing candidate patches for a specific bugid and hunk"""
    return df.loc[(df["bugid"] == bugid) & (df["hunk"] == hunk)]


def get_hunk_candidates(df: pd.DataFrame, hunk: int) -> pd.DataFrame:
    """Returns the subset of `df` containing candidate patches for a specific hunk of a bug"""
    return df.loc[df["hunk"] == hunk]


def get_candidates(df: pd.DataFrame, bugid: str) -> pd.DataFrame:
    """Returns the subset of `df` containing candidate patches for a specific bugid and all its hunks"""
    return df.loc[df["bugid"] == bugid]


def insert_patch(patch, source_file_path, target_file_path, bug_line, bug_len, indent):
    with open(source_file_path, "r") as file:
        lines = file.readlines()
    if bug_len == 0:
        lines.insert(bug_line, indent + patch + "\n")
    else:
        lines[bug_line - 1 : (bug_line - 1) + bug_len] = indent + patch + "\n"

    with open(target_file_path, "w") as file:
        file.writelines(lines)


class Status(Enum):
    PARSABLE = auto()
    UNPARSABLE = auto()


def run_tests(bugid: str, project_dir: Path, filename: str) -> Status:
    with open(project_dir / filename, "rb") as file:
        content = file.read()

    tree = parser.parse(content)
    captures = query.captures(tree.root_node)

    if captures:
        return Status.UNPARSABLE
    else:
        return Status.PARSABLE


def set_plausible(df: pd.DataFrame, bugid: str, hunk: int, patch: str) -> None:
    row_condition = (
        (df["bugid"] == bugid)
        & (df["hunk"] == hunk)
        & (df["decoded_sequences"] == patch)
    )
    df.loc[row_condition, "plausible"] = True


def apply_patch(cp_df: pd.DataFrame, bugid: str, hunks: list) -> Optional[pd.DataFrame]:
    # Load if already processed
    save_file_path = save_state_dir / f"{bugid}.jsonl"
    if save_file_path.exists():
        return pd.read_json(save_file_path, orient="records", lines=True)

    if len(hunks) == 1:
        hunk = hunks[0]

        bug_line, bug_len = hunk["removed_line_numbers_range"]
        bug_hunk_subset_df = get_hunk_candidates(cp_df, 0)

        indent_size = len(hunk["added_lines"]) - len(hunk["added_lines"].lstrip(" \t"))
        indent = hunk["added_lines"][:indent_size]

        filename = Path(hunk["source_path"]).name

        with working_environment(bugid, filename) as (
            project_copy_dir,
            source_file_path,
            target_file_path,
        ):
            for index, patch in bug_hunk_subset_df["decoded_sequences"].items():
                insert_patch(
                    patch, source_file_path, target_file_path, bug_line, bug_len, indent
                )

                # call the testing infrastructure
                start_timer = timeit.default_timer()
                passed = run_tests(bugid, project_copy_dir, filename)
                end_timer = timeit.default_timer()
                cp_df.at[index, "validation_time"] = end_timer - start_timer

                if passed is Status.PARSABLE:
                    cp_df.at[index, "parsable"] = True

        # Save intermediate state
        cp_df.to_json(save_state_dir / f"{bugid}.jsonl", orient="records", lines=True)

        return cp_df

    return None


def copy_dataset_files(dataset_dir, temp_dataset_dir):
    shutil.copytree(
        dataset_dir,
        temp_dataset_dir,
        dirs_exist_ok=False,
        ignore=shutil.ignore_patterns(".*"),
    )


def main():
    n_jobs = 6

    with open(gen_dir / bugs_metadata_file) as meta_file:
        bugs_metadata = ChainMap(*[json.loads(line) for line in meta_file][::-1])

    candidate_patches_df = pd.read_json(
        output_dir / f"final_candidates_{output_size}.jsonl",
        orient="records",
        lines=True,
    )
    candidate_patches_df["parsable"] = False
    candidate_patches_df["validation_time"] = np.nan

    save_state_dir.mkdir(parents=True, exist_ok=True)

    with tqdm_joblib(tqdm(total=len(bugs_metadata))) as progress_bar:
        cp_dfs = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(apply_patch)(
                deepcopy(get_candidates(candidate_patches_df, bugid)), bugid, hunks
            )
            for bugid, hunks in list(bugs_metadata.items())
        )

    concatenated_cp_df = pd.concat(cp_dfs, ignore_index=True)

    bugs_with_plausible_patch = (
        concatenated_cp_df.groupby(["bugid", "hunk"])["parsable"]
        .any()
        .groupby("bugid")
        .all()
    )
    print(bugs_with_plausible_patch)
    print(bugs_with_plausible_patch.value_counts())
    concatenated_cp_df.to_json(
        output_dir / f"plausible_candidates_{output_size}.jsonl",
        orient="records",
        lines=True,
    )


if __name__ == "__main__":
    main()
