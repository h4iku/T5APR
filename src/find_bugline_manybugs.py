"""Finding the buggy line in ManyBugs dataset by comparing buggy and correct versions"""

import difflib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pygments
from pygments.lexers import ClayLexer
from pygments.token import Comment, String
from tqdm import tqdm
from tree_sitter import Language, Parser
from unidiff import PatchSet

from .bugaid_datasets_conf import manybugs_data_dir, manybugs_gen_dir
from .datasets_conf import tree_sitter_lib

Language.build_library(
    str(tree_sitter_lib / "build/my-languages.so"),
    [
        str(tree_sitter_lib / "vendor/tree-sitter-python"),
        str(tree_sitter_lib / "vendor/tree-sitter-java"),
        str(tree_sitter_lib / "vendor/tree-sitter-javascript"),
        str(tree_sitter_lib / "vendor/tree-sitter-c"),
    ],
)

C_LANGUAGE = Language(str(tree_sitter_lib / "build/my-languages.so"), "c")
parser = Parser()
parser.set_language(C_LANGUAGE)
query = C_LANGUAGE.query(
    """
    (function_definition) @function-definition
    """
)

c_lexer = ClayLexer(stripnl=False)


@dataclass
class DiffHunk:
    """Class to keep hunk data"""

    source_path: str
    removed_lines: str
    added_lines: str
    removed_line_numbers_range: tuple[int, int]
    added_line_numbers_range: tuple[int, int]
    source_context: str = ""


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

    lexed_code = pygments.lex(code, c_lexer)

    comment_stripped_code = []
    for ttype, tvalue in lexed_code:
        if (
            ttype in Comment and ttype not in [Comment.Preproc, Comment.PreprocFile]
        ) or ttype in String.Doc:
            newlines = "\n" * (len(tvalue.splitlines()) - 1)
            comment_stripped_code.append(newlines)
        else:
            comment_stripped_code.append(tvalue)

    return "".join(comment_stripped_code)


def get_diff_lines(
    bug_id: str, fromfile: Path, tofile: Path, context_size: int = 0
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
        fromfile=str(fromfile.relative_to(manybugs_data_dir / bug_id / "diffs")),
        tofile=str(tofile.relative_to(manybugs_data_dir / bug_id / "diffs")),
        n=context_size,
    )

    return diff_lines


def process_hunks(diff_lines: Iterable[str]) -> list[DiffHunk]:
    patch_set = PatchSet(diff_lines)

    assert (
        len(patch_set) == 1
    ), "Diffs should only contain one file since I process each file separately"
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

    manybugs_gen_dir.mkdir(parents=True)
    with (
        open(manybugs_gen_dir / "ManyBugs.jsonl", "w", encoding="utf-8") as file,
        open(manybugs_gen_dir / "rem.txt", "w", encoding="utf-8") as remfile,
        open(manybugs_gen_dir / "add.txt", "w", encoding="utf-8") as addfile,
        open(manybugs_gen_dir / "context.txt", "w", encoding="utf-8") as ctxfile,
    ):
        for program, hunks in bug_hunks.items():
            file.write(json.dumps({program: [asdict(h) for h in hunks]}) + "\n")
            remfile.writelines(prepare(h.removed_lines) + "\n" for h in hunks)
            addfile.writelines(prepare(h.added_lines) + "\n" for h in hunks)
            ctxfile.writelines(prepare(h.source_context) + "\n" for h in hunks)


def get_file_path(
    dir_path: Path, modified_file: str, metadata: list[str]
) -> tuple[Path, Path]:
    buggy_file_path = dir_path / "diffs" / f"{modified_file}-{metadata[-2]}"
    fixed_file_path = dir_path / "diffs" / f"{modified_file}-{metadata[-1]}"

    # Handling when the filename uses the complete sha hash
    if not (buggy_file_path.exists() and fixed_file_path.exists()):
        buggy_file_glob = list(buggy_file_path.parent.glob(f"{buggy_file_path.name}*"))
        fixed_file_glob = list(fixed_file_path.parent.glob(f"{fixed_file_path.name}*"))

        assert (
            len(buggy_file_glob) == 1 and len(fixed_file_glob) == 1
        ), "Multiple files matched in the glob"

        buggy_file_path, fixed_file_path = buggy_file_glob[0], fixed_file_glob[0]

    return buggy_file_path, fixed_file_path


def main():
    bug_hunks: dict[str, list[DiffHunk]] = {}

    # TODO: Maybe write some code to handle these cases automatically
    # Some file in ManyBugs don't change
    identicals = {
        "fbc-bug-5556-5557": [
            "./src/rtlib/libfb_qb_str_convto_lng.c",
        ],
        "php-bug-2011-02-01-fefe9fc5c7-0927309852": [
            "ext/phar/phar.c",
        ],
        "php-bug-2012-01-16-f32760bd40-032d140fd6": [
            "Zend/zend_language_scanner.c",
        ],
        "php-bug-2012-03-04-60dfd64bf2-34fe62619d": [
            "Zend/zend_compile.c",
        ],
        "python-bug-69609-69616": [
            "Objects/complexobject.c",
            "Objects/floatobject.c",
            "Python/sysmodule.c",
        ],
        "valgrind-bug-11821-11822": [
            "coregrind/m_gdbserver/m_gdbserver.c",
            "coregrind/m_main.c",
            "coregrind/vgdb.c",
        ],
        "valgrind-bug-12410-12411": [
            "none/tests/rlimit64_nofile.c",
            "none/tests/rlimit_nofile.c",
        ],
        "valgrind-bug-12854-12855": [
            "coregrind/m_main.c",
        ],
    }

    for bug_dir in tqdm(manybugs_data_dir.iterdir()):
        bug_id = bug_dir.name
        metadata = bug_id.split("-")
        with open(bug_dir / "bugged-program.txt") as metafile:
            modified_files = metafile.read().splitlines()

        hunks: list[DiffHunk] = []

        for modified_file in modified_files:
            if bug_id in identicals and modified_file in identicals[bug_id]:
                continue

            buggy_file_path, fixed_file_path = get_file_path(
                bug_dir, modified_file, metadata
            )

            diff_lines = list(get_diff_lines(bug_id, buggy_file_path, fixed_file_path))
            file_hunks = process_hunks(diff_lines)

            for hunk in file_hunks:
                line_number = hunk.removed_line_numbers_range[0]
                hunk.source_context = get_context(buggy_file_path, line_number)

            hunks += file_hunks

        if hunks:
            bug_hunks[bug_id] = hunks

    generate_data(bug_hunks)


if __name__ == "__main__":
    main()
