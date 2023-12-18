from pathlib import Path

from src.bugline_finders.quixbugs_java import (
    get_diff_lines,
    get_program_path,
    process_hunks,
    remove_comments,
)
from src.configs import (
    quixbugs_java_buggy_dir,
    quixbugs_java_correct_dir,
)

fixtures_dir = Path(__file__).parent / "fixtures"


def test_remove_comments():
    """Tests both removing the comments and keeping the line number intact"""

    with open(fixtures_dir / "BUCKETSORT.java") as file:
        code = file.read()

    result = """package java_programs;
import java.util.*;










public class BUCKETSORT {
    public static ArrayList<Integer> bucketsort(ArrayList<Integer> arr, int k) {
        ArrayList<Integer> counts = new ArrayList<Integer>(Collections.nCopies(k,0));
        for (Integer x : arr) {
            counts.set(x,counts.get(x)+1);
        }

        ArrayList<Integer> sorted_arr = new ArrayList<Integer>(100);
	int i = 0;
        for (Integer count : arr) { 
	    sorted_arr.addAll(Collections.nCopies(count, i));
	    i++;
        }

        return sorted_arr;
    }
}
"""
    assert remove_comments(code) == result


def get_processed_hunks(program, buggy_dir, correct_dir):
    buggy_java_program = get_program_path(buggy_dir, program)
    correct_java_program = get_program_path(correct_dir, program)
    diff_lines = get_diff_lines(program, buggy_java_program, correct_java_program)
    hunks = process_hunks(diff_lines)
    return hunks


def test_breadth_first_search_hunks():
    program = "breadth_first_search"
    hunks = get_processed_hunks(
        program, quixbugs_java_buggy_dir, quixbugs_java_correct_dir
    )

    assert len(hunks) == 1
    assert hunks[0].removed_lines.strip() == "while (true) {"
    assert hunks[0].added_lines.strip() == "while (!queue.isEmpty()) {"
    assert hunks[0].removed_line_numbers_range == (24, 1)
    assert hunks[0].added_line_numbers_range == (26, 1)


def test_depth_first_search_hunks():
    program = "depth_first_search"
    hunks = get_processed_hunks(
        program, quixbugs_java_buggy_dir, quixbugs_java_correct_dir
    )

    assert len(hunks) == 1
    assert hunks[0].removed_lines.strip() == ""
    assert hunks[0].added_lines.strip() == "nodesvisited.add(node);"
    assert hunks[0].removed_line_numbers_range == (21, 0)
    assert hunks[0].added_line_numbers_range == (24, 1)


def test_powerset_hunks():
    program = "powerset"
    hunks = get_processed_hunks(
        program, quixbugs_java_buggy_dir, quixbugs_java_correct_dir
    )

    assert len(hunks) == 4


def test_shortest_path_length_hunks():
    program = "shortest_path_length"
    hunks = get_processed_hunks(
        program, quixbugs_java_buggy_dir, quixbugs_java_correct_dir
    )

    assert len(hunks) == 1
    assert (
        hunks[0].removed_lines.strip()
        == "unvisitedNodes.get(nextnode) + length_by_edge.get(Arrays.asList(node, nextnode))));"
    )
    assert (
        hunks[0].added_lines.strip()
        == "distance + length_by_edge.get(Arrays.asList(node, nextnode))));"
    )
    assert hunks[0].removed_line_numbers_range == (38, 1)
    assert hunks[0].added_line_numbers_range == (40, 1)


def test_shortest_paths_hunks():
    program = "shortest_paths"
    hunks = get_processed_hunks(
        program, quixbugs_java_buggy_dir, quixbugs_java_correct_dir
    )

    assert len(hunks) == 1
    assert hunks[0].removed_lines.strip() == "weight_by_edge.put(edge, update_weight);"
    assert (
        hunks[0].added_lines.strip()
        == "weight_by_node.put(edge.get(1), update_weight);"
    )
    assert hunks[0].removed_line_numbers_range == (30, 1)
    assert hunks[0].added_line_numbers_range == (34, 2)


def test_shunting_yard_hunks():
    program = "shunting_yard"
    hunks = get_processed_hunks(
        program, quixbugs_java_buggy_dir, quixbugs_java_correct_dir
    )

    assert len(hunks) == 1
    assert hunks[0].removed_lines.strip() == ""
    assert hunks[0].added_lines.strip() == "opstack.push(token);"
    assert hunks[0].removed_line_numbers_range == (32, 0)
    assert hunks[0].added_line_numbers_range == (34, 1)


def test_sqrt_hunks():
    program = "sqrt"
    hunks = get_processed_hunks(
        program, quixbugs_java_buggy_dir, quixbugs_java_correct_dir
    )

    assert len(hunks) == 1
    assert hunks[0].removed_lines.strip() == "while (Math.abs(x-approx) > epsilon) {"
    assert (
        hunks[0].added_lines.strip() == "while (Math.abs(x-approx*approx) > epsilon) {"
    )
    assert hunks[0].removed_line_numbers_range == (16, 1)
    assert hunks[0].added_line_numbers_range == (16, 1)
