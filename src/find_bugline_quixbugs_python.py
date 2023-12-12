"""Finding the buggy line in QuixBugs(Python) dataset by comparing buggy and correct programs"""

import difflib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pygments
from pygments.filter import Filter, simplefilter
from pygments.lexers import PythonLexer
from pygments.token import Comment, Name, String, Token
from tree_sitter import Language, Parser
from unidiff import PatchSet

from .datasets_conf import (
    quixbugs_genpy_dir,
    quixbugs_programs,
    quixbugs_python_buggy_dir,
    quixbugs_python_correct_dir,
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

PYTHON_LANGUAGE = Language(str(tree_sitter_lib / "build/my-languages.so"), "python")
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)
query = PYTHON_LANGUAGE.query(
    """(function_definition) @func-def
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


python_lexer = PythonLexer(stripnl=False)


class RemCom(Filter):
    def __init__(self, **options):
        Filter.__init__(self, **options)

    def filter(self, lexer, stream):
        for ttype, tvalue in stream:
            if ttype not in Token.Comment and ttype not in Token.Literal.String.Doc:
                yield tvalue


@simplefilter
def rem_com(self, lexer, stream, options):
    for ttype, tvalue in stream:
        if ttype not in Token.Comment and ttype not in Token.Literal.String.Doc:
            yield tvalue


def remove_comments(code: str) -> str:
    """Remove comments and keep the line numbers intact
    so we can replace patched lines in the original file.
    """

    lexed_code = pygments.lex(code, python_lexer)

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
    source_identifiers: list[str] = field(default_factory=list)


def get_program_path(dir_path: Path, program_name: str) -> Path:
    return dir_path / f"{program_name}.py"


def get_diff_lines(
    fromfile: Path, tofile: Path, context_size: int = 0
) -> Iterable[str]:
    with open(fromfile) as source_file, open(tofile) as target_file:
        source = remove_comments(source_file.read())
        target = remove_comments(target_file.read())

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

        # ignore hunks where both source and target are empty
        if not (hunk_source.strip() or hunk_target.strip()):
            continue

        # ignore if hunks only differ in trailing whitespaces
        if hunk_source.rstrip() == hunk_target.rstrip():
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

    quixbugs_genpy_dir.mkdir(parents=True)
    with (
        open(quixbugs_genpy_dir / "QuixBugs_Python.jsonl", "w") as file,
        open(quixbugs_genpy_dir / "rem.txt", "w") as remfile,
        open(quixbugs_genpy_dir / "add.txt", "w") as addfile,
        open(quixbugs_genpy_dir / "context.txt", "w") as ctxfile,
    ):
        for program, hunks in programs_hunks.items():
            file.write(json.dumps({program: [asdict(h) for h in hunks]}) + "\n")
            remfile.writelines(prepare(h.removed_lines) + "\n" for h in hunks)
            addfile.writelines(prepare(h.added_lines) + "\n" for h in hunks)
            ctxfile.writelines(prepare(h.source_context) + "\n" for h in hunks)


def get_identifiers(context: str) -> list[str]:
    lexed_code = pygments.lex(context, python_lexer)
    identifiers = {tvalue for ttype, tvalue in lexed_code if ttype in Name}
    return list(identifiers)


def main():
    programs_hunks: dict[str, list[DiffHunk]] = {}

    for program in quixbugs_programs:
        buggy_python_program = get_program_path(quixbugs_python_buggy_dir, program)
        correct_python_program = get_program_path(quixbugs_python_correct_dir, program)

        diff_lines = list(get_diff_lines(buggy_python_program, correct_python_program))
        hunks = process_hunks(diff_lines)

        # if Quixbugs(Python)
        assert len(hunks) == 1, "QuixBugs Python programs should all have one hunk"

        for hunk in hunks:
            line_number = hunk.removed_line_numbers_range[0]
            hunk.source_context = get_context(buggy_python_program, line_number)
            # hunk.source_identifiers = get_identifiers(hunk.source_context)

        programs_hunks[program] = hunks

    generate_data(programs_hunks)


if __name__ == "__main__":
    main()
