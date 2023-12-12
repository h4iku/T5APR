"""Finding the buggy line in QuixBugs(Java) dataset by comparing buggy and correct programs"""

import difflib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pygments
from pygments.lexers import JavaLexer
from pygments.token import Comment, Name, String
from tree_sitter import Language, Parser
from unidiff import PatchSet

from .datasets_conf import (
    quixbugs_genjava_dir,
    quixbugs_java_buggy_dir,
    quixbugs_java_correct_dir,
    quixbugs_programs,
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


def get_context(program: Path, line_number: int) -> str:
    with open(program, "rb") as file:
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


java_lexer = JavaLexer(stripnl=False)


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


@dataclass
class DiffHunk:
    """Class to keep hunk data"""

    removed_lines: str
    added_lines: str
    removed_line_numbers_range: tuple[int, int]
    added_line_numbers_range: tuple[int, int]
    source_context: str = ""


def get_program_path(dir_path: Path, program_name: str) -> Path:
    return dir_path / f"{program_name.upper()}.java"


def get_diff_lines(
    program: str, fromfile: Path, tofile: Path, context_size: int = 0
) -> Iterable[str]:
    with open(fromfile) as source_file, open(tofile) as target_file:
        source, target = cleanup(program, source_file.read(), target_file.read())

    source = remove_comments(source)
    target = remove_comments(target)

    diff_lines = difflib.unified_diff(
        source.splitlines(keepends=True),
        target.splitlines(keepends=True),
        fromfile="/".join(fromfile.parts[-2:]),
        tofile="/".join(tofile.parts[-2:]),
        n=context_size,
    )

    return diff_lines


def process_hunks(diff_lines: Iterable[str]) -> list[DiffHunk]:
    patch_set = PatchSet(diff_lines)
    assert len(patch_set) == 1
    patched_file = patch_set[0]

    diff_hunks = []

    for hunk in patched_file:
        hunk_source = "".join(x[1:] for x in hunk.source)
        hunk_target = "".join(x[1:] for x in hunk.target)

        # Ignore if hunks only differ in whitespaces
        if "".join(hunk_source.split()) == "".join(hunk_target.split()):
            continue

        # Ignore if the difference is only in the package name
        if hunk_source.strip().startswith("package") and hunk_target.strip().startswith(
            "package"
        ):
            continue

        # Ignore if the difference is only in the imports
        if hunk_source.strip().startswith("import") or hunk_target.strip().startswith(
            "import"
        ):
            continue

        diff_hunk = DiffHunk(
            hunk_source,
            hunk_target,
            (hunk.source_start, hunk.source_length),
            (hunk.target_start, hunk.target_length),
        )
        diff_hunks.append(diff_hunk)

    return diff_hunks


def generate_data(programs_hunks: dict[str, list[DiffHunk]]) -> None:
    """Generates input data for model evaluation"""

    def prepare(hunk: str) -> str:
        lines_concat = " ".join([line.strip() for line in hunk.splitlines()])
        return lines_concat.strip()

    quixbugs_genjava_dir.mkdir(parents=True)
    with (
        open(quixbugs_genjava_dir / "QuixBugs_Java.jsonl", "w") as file,
        open(quixbugs_genjava_dir / "rem.txt", "w") as remfile,
        open(quixbugs_genjava_dir / "add.txt", "w") as addfile,
        open(quixbugs_genjava_dir / "context.txt", "w") as ctxfile,
    ):
        for program, hunks in programs_hunks.items():
            file.write(json.dumps({program: [asdict(h) for h in hunks]}) + "\n")
            remfile.writelines(prepare(h.removed_lines) + "\n" for h in hunks)
            addfile.writelines(prepare(h.added_lines) + "\n" for h in hunks)
            ctxfile.writelines(prepare(h.source_context) + "\n" for h in hunks)


def get_identifiers(context: str) -> list[str]:
    lexed_code = pygments.lex(context, java_lexer)
    identifiers = {tvalue for ttype, tvalue in lexed_code if ttype in Name}
    return list(identifiers)


def cleanup(program: str, source: str, target: str) -> tuple[str, str]:
    """Clean up inconsistencies in some files to detect changes more accurately"""

    if program == "breadth_first_search":
        source = source.replace("// return false;", "return false;")
    elif program == "shortest_paths":
        source_lines = source.splitlines(keepends=True)
        source = "".join(source_lines[:36] + source_lines[62:])
    elif program == "shunting_yard":
        source_lines = source.splitlines(keepends=True)
        del source_lines[41]
        source = "".join(source_lines)

    return source, target


def main():
    programs_hunks: dict[str, list[DiffHunk]] = {}

    for program in quixbugs_programs:
        buggy_java_program = get_program_path(quixbugs_java_buggy_dir, program)
        correct_java_program = get_program_path(quixbugs_java_correct_dir, program)

        diff_lines = list(
            get_diff_lines(program, buggy_java_program, correct_java_program)
        )
        hunks = process_hunks(diff_lines)

        for hunk in hunks:
            line_number = hunk.removed_line_numbers_range[0]
            hunk.source_context = get_context(buggy_java_program, line_number)

        programs_hunks[program] = hunks

    generate_data(programs_hunks)


if __name__ == "__main__":
    main()
