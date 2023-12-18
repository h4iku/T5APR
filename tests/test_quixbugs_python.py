from pathlib import Path

from src.bugline_finders.quixbugs_python import (
    get_context,
    get_diff_lines,
    process_hunks,
    remove_comments,
)

fixtures_dir = Path(__file__).parent / "fixtures"


def test_remove_comments():
    """Tests both removing the comments and keeping the line number intact"""

    code = '''
"""module docstring"""
# Python 3
def possible_change(coins, total):
    """function docstring"""
    if total == 0:
        return 1  # An inline comment
    if total < 0:
        return 0

    string = """some random string"""


"""
Making Change
change

"""

"""another string"""
'''
    result = '''


def possible_change(coins, total):
    
    if total == 0:
        return 1  
    if total < 0:
        return 0

    string = """some random string"""









'''
    assert remove_comments(code) == result


def test_get_diff_lines():
    source_path = fixtures_dir / "buggy_detect_cycle.py"
    target_path = fixtures_dir / "correct_detect_cycle.py"
    result = """--- fixtures/buggy_detect_cycle.py
+++ fixtures/correct_detect_cycle.py
@@ -5 +5 @@
-        if hare.successor is None:
+        if hare is None or hare.successor is None:
@@ -27,0 +28 @@
+
""".splitlines(keepends=True)
    assert list(get_diff_lines(source_path, target_path)) == result


def test_process_hunks():
    diff_lines = """--- python_programs/bitcount.py
+++ correct_python_programs/bitcount.py
@@ -5,1 +5,1 @@
-        n ^= n - 1
+        n &= n - 1
@@ -6,1 +6,1 @@
-        n ^= n - 1  
+        n ^= n - 1
@@ -8,3 +7,0 @@
-
-
-
""".splitlines(keepends=True)
    diff_hunks = process_hunks(diff_lines)
    assert len(diff_hunks) == 1
    assert diff_hunks[0].removed_lines == "        n ^= n - 1\n"
    assert diff_hunks[0].added_lines == "        n &= n - 1\n"


def test_get_context():
    source_path = fixtures_dir / "buggy_detect_cycle.py"
    result = """def detect_cycle(node):
    hare = tortoise = node

    while True:
        if hare.successor is None:
            return False

        tortoise = tortoise.successor
        hare = hare.successor.successor

        if hare is tortoise:
            return True"""

    assert get_context(source_path, 4) == result
