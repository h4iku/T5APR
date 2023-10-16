"""Finding the buggy line in Codeflaws dataset by comparing buggy and correct versions"""

import csv
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

from .bugaid_datasets_conf import codeflaws_data_dir, codeflaws_gen_dir
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
            # C lexer doesn't detect the extra newline token at the end of single comments:
            # https://github.com/pygments/pygments/issues/2284
            # So we use another C-like lexer
            # -1 is because there is a newline at the end of comment tokens
            newlines = "\n" * (len(tvalue.splitlines()) - 1)
            comment_stripped_code.append(newlines)
        else:
            comment_stripped_code.append(tvalue)

    return "".join(comment_stripped_code)


def get_diff_lines(
    bug_id: str, fromfile: Path, tofile: Path, context_size: int = 0
) -> Iterable[str]:
    with (
        open(fromfile, encoding="utf-8") as source_file,
        open(tofile, encoding="utf-8") as target_file,
    ):
        source = remove_comments(source_file.read())
        target = remove_comments(target_file.read())

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

        # Ignore hunks where both source and target are empty
        if not (hunk_source.strip() or hunk_target.strip()):
            continue

        # Ignore if hunks only differ in trailing whitespaces
        if hunk_source.strip() == hunk_target.strip():
            continue

        # Ignore if hunks only differ in whitespaces
        # if "".join(hunk_source.split()) == "".join(hunk_target.split()):
        #     continue

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

    with (
        open(codeflaws_gen_dir / "Codeflaws.jsonl", "w", encoding="utf-8") as file,
        open(codeflaws_gen_dir / "rem.txt", "w", encoding="utf-8") as remfile,
        open(codeflaws_gen_dir / "add.txt", "w", encoding="utf-8") as addfile,
        open(codeflaws_gen_dir / "context.txt", "w", encoding="utf-8") as ctxfile,
    ):
        for program, hunks in bug_hunks.items():
            file.write(json.dumps({program: [asdict(h) for h in hunks]}) + "\n")
            remfile.writelines(prepare(h.removed_lines) + "\n" for h in hunks)
            addfile.writelines(prepare(h.added_lines) + "\n" for h in hunks)
            ctxfile.writelines(prepare(h.source_context) + "\n" for h in hunks)


def get_file_path(dir_path: Path, file_name: str) -> Path:
    return dir_path / f"{file_name}.c"


def main():
    bug_hunks: dict[str, list[DiffHunk]] = {}

    # Some bugs in Codeflaws have same buggy and fixed programs
    identical_bug_ids = [
        "71-A-bug-18359456-18359477",
        "558-B-bug-12585906-12585920",
        "382-A-bug-8368809-8368827",
        "569-C-bug-12481867-12481905",
        "289-D-bug-3473596-3473601",
        "289-D-bug-3473592-3473601",
        "431-C-bug-15194556-15194577",
    ]

    with open(
        codeflaws_data_dir / "codeflaws-defect-detail-info.txt", newline=""
    ) as metafile:
        reader = csv.reader(metafile, delimiter="\t")

        for row in tqdm(reader):
            bug_id = row[0]

            if bug_id in identical_bug_ids:
                continue

            metadata = row[0].split("-")
            buggy_file_name = f"{metadata[0]}-{metadata[1]}-{metadata[-2]}"
            fixed_file_name = f"{metadata[0]}-{metadata[1]}-{metadata[-1]}"

            hunks: list[DiffHunk] = []

            buggy_file_path = get_file_path(
                codeflaws_data_dir / bug_id, buggy_file_name
            )
            fixed_file_path = get_file_path(
                codeflaws_data_dir / bug_id, fixed_file_name
            )

            diff_lines = list(get_diff_lines(bug_id, buggy_file_path, fixed_file_path))
            hunks = process_hunks(diff_lines)

            for hunk in hunks:
                line_number = hunk.removed_line_numbers_range[0]
                hunk.source_context = get_context(buggy_file_path, line_number)

            if bug_id == "230-B-bug-16820314-16820335":
                hunks = hunks[0:1]
            elif bug_id == "474-A-bug-18022668-18022684":
                hunks = hunks[1:2]

            bug_hunks[bug_id] = hunks

        generate_data(bug_hunks)


if __name__ == "__main__":
    main()
