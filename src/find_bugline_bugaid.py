"""Finding the buggy line in BugAID dataset by comparing buggy and correct versions"""

import csv
import difflib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pygments
from pygments.lexers import JavascriptLexer
from pygments.token import Comment, String
from tree_sitter import Language, Parser
from unidiff import PatchSet

from .bugaid_datasets_conf import bugaid_data_dir, bugaid_gen_dir
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

JAVASCRIPT_LANGUAGE = Language(
    str(tree_sitter_lib / "build/my-languages.so"), "javascript"
)
parser = Parser()
parser.set_language(JAVASCRIPT_LANGUAGE)
query = JAVASCRIPT_LANGUAGE.query(
    """
    (function) @function
    (function_declaration) @function-declaration
    """
)

javascript_lexer = JavascriptLexer(stripnl=False)


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

    lexed_code = pygments.lex(code, javascript_lexer)

    comment_stripped_code = []
    for ttype, tvalue in lexed_code:
        if ttype in Comment or ttype in String.Doc:
            # -1 is because there is a newline at the end of comment tokens
            comment_stripped_code.append("\n" * (len(tvalue.splitlines()) - 1))
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

        source, target = cleanup(bug_id, source, target)

        source_lines = [line.strip() + "\n" for line in source.splitlines()]
        target_lines = [line.strip() + "\n" for line in target.splitlines()]

    diff_lines = difflib.unified_diff(
        source_lines,
        target_lines,
        fromfile="/".join(fromfile.parts[-2:]),
        tofile="/".join(tofile.parts[-2:]),
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

    bugaid_gen_dir.mkdir(parents=True)
    with (
        open(bugaid_gen_dir / "BugAid.jsonl", "w") as file,
        open(bugaid_gen_dir / "rem.txt", "w") as remfile,
        open(bugaid_gen_dir / "add.txt", "w") as addfile,
        open(bugaid_gen_dir / "context.txt", "w") as ctxfile,
    ):
        for program, hunks in bug_hunks.items():
            file.write(json.dumps({program: [asdict(h) for h in hunks]}) + "\n")
            remfile.writelines(prepare(h.removed_lines) + "\n" for h in hunks)
            addfile.writelines(prepare(h.added_lines) + "\n" for h in hunks)
            ctxfile.writelines(prepare(h.source_context) + "\n" for h in hunks)


def cleanup(program: str, source: str, target: str) -> tuple[str, str]:
    """Clean up inconsistencies in some files to detect changes more accurately"""

    if program == "IncorrectComparison1":
        source = source.replace(
            "if (typeof opt.default!='undefined') self.default(key, opt.default);",
            """if (typeof opt.default!='undefined') {
                self.default(key, opt.default);
            }""",
        )

    return source, target


def main():
    bug_hunks: dict[str, list[DiffHunk]] = {}

    with open(bugaid_data_dir.parent / "metadata.txt", newline="") as metafile:
        reader = csv.reader(metafile)
        bug_ids, file_names = zip(*[row for row in reader])

    for bug_id, file_name in zip(bug_ids, file_names):
        hunks: list[DiffHunk] = []

        buggy_file_path = bugaid_data_dir / bug_id / "buggy" / file_name
        fixed_file_path = bugaid_data_dir / bug_id / "fixed" / file_name

        diff_lines = list(get_diff_lines(bug_id, buggy_file_path, fixed_file_path))
        hunks = process_hunks(diff_lines)

        for hunk in hunks:
            line_number = hunk.removed_line_numbers_range[0]
            hunk.source_context = get_context(buggy_file_path, line_number)

        bug_hunks[bug_id] = hunks

    generate_data(bug_hunks)


if __name__ == "__main__":
    main()
