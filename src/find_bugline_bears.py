"""Finding the buggy line in Bears benchmark by comparing buggy and correct versions"""


import difflib
import json

import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import contextlib
import joblib

import pygments
from joblib import Parallel, delayed
from pygments.lexers import JavaLexer
from pygments.token import Comment, String
from tqdm import tqdm
from tree_sitter import Language, Parser
from unidiff import PatchSet

from d4j_datasets_conf import bears_root, bears_gen_dir, bears_tmp_dir, tree_sitter_lib


Language.build_library(
    str(tree_sitter_lib / "build/my-languages.so"),
    [
        str(tree_sitter_lib / "vendor/tree-sitter-python"),
        str(tree_sitter_lib / "vendor/tree-sitter-java"),
        str(tree_sitter_lib / "vendor/tree-sitter-javascript"),
        str(tree_sitter_lib / "vendor/tree-sitter-c"),
    ],
)

JAVA_LANGUAGE = Language(str(tree_sitter_lib / "build/my-languages.so"), "java")
parser = Parser()
parser.set_language(JAVA_LANGUAGE)
query = JAVA_LANGUAGE.query(
    """(method_declaration) @method-dec
    """
)

java_lexer = JavaLexer(stripnl=False)


@dataclass
class DiffHunk:
    """Class to keep hunk data"""

    source_path: str
    removed_lines: str
    added_lines: str
    removed_line_numbers_range: tuple[int, int]
    added_line_numbers_range: tuple[int, int]
    source_context: str = ""


@contextlib.contextmanager
def working_environment(bug_id: str, branch: str, diff: str):
    buggy_checkout_dir = bears_tmp_dir / f"{bug_id}-buggy"
    fixed_checkout_dir = bears_tmp_dir / f"{bug_id}-fixed"

    # Checkout buggy and fixed versions of the source code
    checkout_source(branch, True, buggy_checkout_dir)
    checkout_source(branch, False, fixed_checkout_dir)

    modified_files = [f.path for f in PatchSet(diff).modified_files]

    try:
        yield buggy_checkout_dir, fixed_checkout_dir, modified_files
    finally:
        shutil.rmtree(buggy_checkout_dir)
        shutil.rmtree(fixed_checkout_dir)
        run_cmd(f"git -C {bears_root} worktree prune")


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


def get_context(file_path: Path, line_number: int) -> str:
    with open(file_path, "rb") as file:
        content = file.read()

    tree = parser.parse(content)

    captures = query.captures(tree.root_node)

    for capture, _ in captures:
        # start_point and end_point are zero-based
        if capture.start_point[0] <= (line_number - 1) <= capture.end_point[0]:
            context = capture.text.decode()
            # splitlines and join are to handle Python's universal newlines
            # conversion on Windows to avoid getting \r\r\n
            return "\n".join(context.splitlines())
    else:
        return ""


def remove_comments(code: str) -> str:
    """Remove comments and keep the line numbers intact
    so we can replace patched lines in the original file.
    """

    lexed_code = pygments.lex(code, java_lexer)

    comment_stripped_code = []
    for ttype, tvalue in lexed_code:
        if ttype in Comment or ttype in String.Doc:
            # -1 is because there is a newline at the end of comment tokens
            comment_stripped_code.append("\n" * (len(tvalue.splitlines()) - 1))
        else:
            comment_stripped_code.append(tvalue)

    return "".join(comment_stripped_code)


def get_diff_lines(
    modified_file_path: str, fromfile: Path, tofile: Path, context_size: int = 0
) -> Iterable[str]:
    with (
        open(fromfile, encoding="cp1256") as source_file,
        open(tofile, encoding="cp1256") as target_file,
    ):
        source = remove_comments(source_file.read())
        target = remove_comments(target_file.read())

    source_lines = [line.strip() + "\n" for line in source.splitlines()]
    target_lines = [line.strip() + "\n" for line in target.splitlines()]

    diff_lines = difflib.unified_diff(
        source_lines,
        target_lines,
        fromfile=modified_file_path,
        tofile=modified_file_path,
        n=context_size,
    )

    return diff_lines


def process_hunks(diff_lines: Iterable[str]) -> list[DiffHunk]:
    patch_set = PatchSet(diff_lines)

    # My diffs should only contain one file since I process each file separately
    assert len(patch_set) == 1
    patched_file = patch_set[0]

    diff_hunks = []

    for hunk in patched_file:
        hunk_source = "".join(x[1:] for x in hunk.source)
        hunk_target = "".join(x[1:] for x in hunk.target)

        # Ignore if hunks only differ in whitespaces
        if "".join(hunk_source.split()) == "".join(hunk_target.split()):
            continue

        diff_hunk = DiffHunk(
            patched_file.source_file,
            hunk_source,
            hunk_target,
            (hunk.source_start, hunk.source_length),
            (hunk.target_start, hunk.target_length),
        )
        diff_hunks.append(diff_hunk)

    return diff_hunks


def generate_data(bug_hunks: dict[str, list[DiffHunk]]) -> None:
    """Generates input data for model evaluation"""

    def prepare(hunk: str) -> str:
        lines_concat = " ".join([line.strip() for line in hunk.splitlines()])
        return lines_concat.strip()

    bears_gen_dir.mkdir(parents=True)

    with (
        open(bears_gen_dir / "Bears.jsonl", "w") as file,
        open(bears_gen_dir / "rem.txt", "w") as remfile,
        open(bears_gen_dir / "add.txt", "w") as addfile,
        open(bears_gen_dir / "context.txt", "w") as ctxfile,
    ):
        for program, hunks in bug_hunks.items():
            file.write(json.dumps({program: [asdict(h) for h in hunks]}) + "\n")
            remfile.writelines(prepare(h.removed_lines) + "\n" for h in hunks)
            addfile.writelines(prepare(h.added_lines) + "\n" for h in hunks)
            ctxfile.writelines(prepare(h.source_context) + "\n" for h in hunks)


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    args = shlex.split(cmd)
    result = subprocess.run(args, capture_output=True, text=True, check=check)
    return result


# FIXME: Maybe put this inside the working_environment context manager
def checkout_source(branch: str, buggy: bool, checkout_dir: Path) -> None:
    cmd = f"git -C {bears_root} worktree add {checkout_dir} {branch}"
    run_cmd(cmd)

    commits_count = int(
        run_cmd(f"git -C {checkout_dir} rev-list --count HEAD").stdout.strip()
    )
    assert commits_count in [
        3,
        4,
    ], f"Should be three or four commits in each branch, {checkout_dir}"

    if buggy:
        run_cmd(f"git -C {checkout_dir} checkout HEAD~2")

    (checkout_dir / ".git").unlink()


def worker_func(bug: dict[str, Any]) -> tuple[str, list[DiffHunk]]:
    """Worker function to process each bug"""

    hunks: list[DiffHunk] = []

    with working_environment(bug["bugId"], bug["bugName"], bug["diff"]) as (
        buggy_checkout_dir,
        fixed_checkout_dir,
        modified_files,
    ):
        for modified_file in modified_files:
            buggy_file_path = buggy_checkout_dir / modified_file
            fixed_file_path = fixed_checkout_dir / modified_file

            # if source or target doesn't exist, patch needs creation or deletion of a file
            if buggy_file_path.exists() and fixed_file_path.exists():
                diff_lines = list(
                    get_diff_lines(modified_file, buggy_file_path, fixed_file_path)
                )
                file_hunks = process_hunks(diff_lines)

                for hunk in file_hunks:
                    line_number = hunk.removed_line_numbers_range[0]
                    hunk.source_context = get_context(buggy_file_path, line_number)

                hunks += file_hunks

    return bug["bugId"], hunks


def main():
    n_jobs = 6
    bug_hunks: dict[str, list[DiffHunk]] = {}

    bears_bugs_meta = bears_root / "docs/data/bears-bugs.json"
    with open(bears_bugs_meta) as meta_file:
        bears_bugs: list[dict[str, Any]] = json.load(meta_file)

    with tqdm_joblib(tqdm(total=len(bears_bugs))) as progress_bar:
        result = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(worker_func)(bug) for bug in bears_bugs
        )

    for bug_id, hunks in result:
        bug_hunks[bug_id] = hunks

    generate_data(bug_hunks)

    print("Generating done! Cleaning temp...")
    # only delete directories that start with `Bears`
    for directory in bears_tmp_dir.iterdir():
        if directory.name.startswith("Bears"):
            shutil.rmtree(directory)


if __name__ == "__main__":
    main()
