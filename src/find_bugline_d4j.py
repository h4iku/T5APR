"""Finding the buggy line in Defects4J dataset by comparing buggy and correct versions"""

import difflib
import json
import re
import shlex
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import contextlib
import joblib

import pygments
from joblib import Parallel, delayed
from pygments.lexers import JavaLexer
from pygments.token import Comment, String
from tqdm import tqdm
from tree_sitter import Language, Parser
from unidiff import PatchSet

from d4j_datasets_conf import (
    d4j_bin,
    d4j_gen_dir,
    d4j_tmp_dir,
    d4j_version,
    tree_sitter_lib,
)


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


def check_java_version():
    try:
        java_version_string = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, check=True
        ).stderr
    except subprocess.CalledProcessError as e:
        print("Can't find `java`")

    pattern = '"(\d+\.\d+).*"'
    java_version = re.search(pattern, java_version_string).groups()[0]

    if d4j_version == "1.4.0":
        assert java_version == "1.7", "Wrong Java version, needs Java 7"
    elif d4j_version == "2.0.0":
        assert java_version == "1.8", "Wrong Java version, needs Java 8"


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
    pid: int, fromfile: Path, tofile: Path, context_size: int = 0
) -> Iterable[str]:
    with (
        open(fromfile, encoding="cp1256") as source_file,
        open(tofile, encoding="cp1256") as target_file,
    ):
        source = remove_comments(source_file.read())
        target = remove_comments(target_file.read())

    diff_lines = difflib.unified_diff(
        source.splitlines(keepends=True),
        target.splitlines(keepends=True),
        fromfile=str(fromfile.relative_to(d4j_tmp_dir / f"{pid}/buggy")),
        tofile=str(tofile.relative_to(d4j_tmp_dir / f"{pid}/fixed")),
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

        # Ignore hunks where both source and target are empty
        if not (hunk_source.strip() or hunk_target.strip()):
            continue

        # Ignore if hunks only differ in trailing whitespaces
        if hunk_source.strip() == hunk_target.strip():
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

    d4j_gen_dir.mkdir(parents=True)

    with (
        open(d4j_gen_dir / "Defects4J.jsonl", "w") as file,
        open(d4j_gen_dir / "rem.txt", "w") as remfile,
        open(d4j_gen_dir / "add.txt", "w") as addfile,
        open(d4j_gen_dir / "context.txt", "w") as ctxfile,
    ):
        for program, hunks in bug_hunks.items():
            file.write(json.dumps({program: [asdict(h) for h in hunks]}) + "\n")
            remfile.writelines(prepare(h.removed_lines) + "\n" for h in hunks)
            addfile.writelines(prepare(h.added_lines) + "\n" for h in hunks)
            ctxfile.writelines(prepare(h.source_context) + "\n" for h in hunks)


def get_file_path(dir_path: Path, class_name: str) -> Path:
    return dir_path / f"{class_name.replace('.', '/')}.java"


def run_d4j_cmd(cmd: str) -> str:
    d4j_cmd = f"perl {d4j_bin} {cmd}"
    args = shlex.split(d4j_cmd)
    result = subprocess.run(args, capture_output=True, check=True, text=True)
    return result.stdout


def run_project_query(project_id: str, fields: str) -> str:
    cmd = f"query -p {project_id} -q {fields}"
    return run_d4j_cmd(cmd)


def checkout_source(
    project_id: str, bug_id: str, buggy: bool, checkout_dir: Path
) -> None:
    checkout_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"checkout -p {project_id} -v {bug_id}{'b' if buggy else 'f'} -w {checkout_dir}"
    )
    run_d4j_cmd(cmd)


def worker_func(project_id: str, bug_id: str) -> tuple[str, list[DiffHunk]]:
    """Worker function to process each project bugs"""

    pid = threading.get_ident()

    buggy_checkout_dir = d4j_tmp_dir / f"{pid}/buggy"
    fixed_checkout_dir = d4j_tmp_dir / f"{pid}/fixed"

    # Checkout buggy and fixed versions of the source code
    checkout_source(project_id, bug_id, True, buggy_checkout_dir)
    checkout_source(project_id, bug_id, False, fixed_checkout_dir)
    source_dir_name = run_d4j_cmd(f"export -p dir.src.classes -w {buggy_checkout_dir}")
    modified_classes: list[str] = run_d4j_cmd(
        f"export -p classes.modified -w {buggy_checkout_dir}"
    ).splitlines()

    hunks: list[DiffHunk] = []

    for modified_class in modified_classes:
        buggy_file_path = get_file_path(
            buggy_checkout_dir / source_dir_name, modified_class
        )
        fixed_file_path = get_file_path(
            fixed_checkout_dir / source_dir_name, modified_class
        )

        # if source or target doesn't exist, patch needs creation or deletion of a file
        if buggy_file_path.exists() and fixed_file_path.exists():
            diff_lines = list(get_diff_lines(pid, buggy_file_path, fixed_file_path))
            file_hunks = process_hunks(diff_lines)

            for hunk in file_hunks:
                line_number = hunk.removed_line_numbers_range[0]
                hunk.source_context = get_context(buggy_file_path, line_number)

            hunks += file_hunks

    return bug_id, hunks


def main():
    check_java_version()

    n_jobs = 6
    bug_hunks: dict[str, list[DiffHunk]] = {}

    projects: dict[str, list[str]] = {}
    if d4j_version == "2.0.0":
        projects = {
            pid: run_d4j_cmd(f"bids -p {pid}").splitlines()
            for pid in run_d4j_cmd("pids").splitlines()
        }
    else:
        projects = {
            "Closure": ["63", "93"],
            "Lang": ["2"],
            "Time": ["21"],
        }

    for project_id, bug_ids in projects.items():
        print(project_id)

        with tqdm_joblib(tqdm(total=len(bug_ids))) as progress_bar:
            result = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(worker_func)(project_id, bug_id) for bug_id in bug_ids
            )

        for bug_id, hunks in result:
            bug_hunks[f"{project_id} {bug_id}"] = hunks

    generate_data(bug_hunks)

    print("Generating done! Cleaning temp...")
    # only delete directories in the form of `<int>/[buggy|fixed]`
    for directory in d4j_tmp_dir.iterdir():
        if directory.name.isdecimal():
            shutil.rmtree(directory)


if __name__ == "__main__":
    main()
