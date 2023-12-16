import contextlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import timeit
from collections import ChainMap, defaultdict
from copy import deepcopy
from enum import Enum, auto
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib
import numpy as np
import pandas as pd
import xmltodict
from d4j_datasets_conf import bears_gen_dir, bears_root
from joblib import Parallel, delayed
from tqdm import tqdm

gen_dir = bears_gen_dir
bugs_metadata_file = "Bears.jsonl"
output_dir = gen_dir / "outputs-multi"
# output_dir = gen_dir / "outputs-java"
bears_tmp_dir = output_dir / "temp"
save_state_dir = output_dir / "save-state"
output_size = 100

java8_home_trava = Path.home() / ".sdkman/candidates/java/8.0.232-trava"
java8_home_tem = Path.home() / ".sdkman/candidates/java/8.0.345-tem"

rem_file_path = gen_dir / "rem.txt"
add_file_path = gen_dir / "add.txt"
with (
    open(rem_file_path) as rem_file,
    open(add_file_path) as add_file,
):
    sources = [src.strip() for src in rem_file]
    targets = [tgt.strip() for tgt in add_file]

bears_bugs_meta = bears_root / "docs/data/bears-bugs.json"
with open(bears_bugs_meta) as meta_file:
    bears_bugs: list[dict[str, Any]] = json.load(meta_file)


@contextlib.contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(newdir.expanduser())
    try:
        yield
    finally:
        os.chdir(prevdir)


def process_projects(bugid: str, checkout_dir: Path, pom_path: str) -> None:
    # TODO: I should extract common patterns from this function and create new functions.

    proj_pom_path = checkout_dir / pom_path

    def read_pom():
        with open(proj_pom_path) as pom_file:
            pom_dict = xmltodict.parse(
                pom_file.read(),
                force_list={"exclude", "plugin", "repository", "pluginRepository"},
            )
        return pom_dict

    def write_pom(pom_dict):
        with open(proj_pom_path, "w") as pom_file:
            pom_file.write(xmltodict.unparse(pom_dict, pretty=True))

    def http_repo_to_https(pom_dict):
        for repo in pom_dict["project"]["repositories"]["repository"]:
            if "http:" in repo["url"]:
                repo["url"] = repo["url"].replace("http:", "https:")
        return pom_dict

    # These are the FasterXML-jackson-databind project
    if bugid in ["Bears-3", "Bears-8", "Bears-9", "Bears-19", "Bears-22", "Bears-25"]:
        pom_dict = read_pom()
        content = "com/fasterxml/jackson/databind/ext/TestCoreXMLTypes.java"
        for plugin in pom_dict["project"]["build"]["plugins"]["plugin"]:
            if plugin["artifactId"] == "maven-surefire-plugin":
                plugin["configuration"]["excludes"]["exclude"].append(content)
        write_pom(pom_dict)

    # These are the INRIA-spoon project
    elif bugid in [
        "Bears-27",
        "Bears-32",
        "Bears-36",
        "Bears-38",
        "Bears-42",
        "Bears-43",
        "Bears-46",
        "Bears-47",
        "Bears-49",
        "Bears-53",
        "Bears-56",
        "Bears-61",
        "Bears-74",
        "Bears-76",
        "Bears-78",
    ]:
        pom_dict = read_pom()

        exclusion = {
            "exclusions": {
                "exclusion": {"groupId": "commons-io", "artifactId": "commons-io"}
            }
        }
        for dependency in pom_dict["project"]["dependencies"]["dependency"]:
            if dependency["groupId"] == "com.github.stefanbirkner":
                dependency.update(exclusion)

        excludes = {
            "excludes": {
                "exclude": "spoon/reflect/declaration/CtTypeInformationTest.java"
            }
        }
        for plugin in pom_dict["project"]["build"]["plugins"]["plugin"]:
            if plugin["artifactId"] == "maven-surefire-plugin":
                plugin["configuration"].update(excludes)

        write_pom(pom_dict)

    elif bugid in ["Bears-85", "Bears-87", "Bears-88", "Bears-89"]:
        pom_dict = read_pom()

        new_repo = {
            "id": "com.springsource.repository.bundles.release",
            "url": "https://repository.springsource.com/maven/bundles/release",
        }
        pom_dict["project"]["pluginRepositories"]["pluginRepository"].append(new_repo)

        write_pom(pom_dict)

    elif bugid in ["Bears-157"]:
        pom_dict = read_pom()

        excludes = {
            "excludes": {
                "exclude": ["**/NettyCommTest.java", "**/ManagementServerTest.java"]
            }
        }

        for plugin in pom_dict["project"]["build"]["plugins"]["plugin"]:
            if plugin["artifactId"] == "maven-surefire-plugin":
                plugin["configuration"].update(excludes)

        for repo in pom_dict["project"]["repositories"]["repository"]:
            if "http:" in repo["url"]:
                repo["url"] = repo["url"].replace("http:", "https:")

        write_pom(pom_dict)

    elif bugid in ["Bears-159", "Bears-160", "Bears-162", "Bears-163"]:
        pom_dict = read_pom()
        pom_dict = http_repo_to_https(pom_dict)
        write_pom(pom_dict)

    elif bugid in ["Bears-165"]:
        pom_dict = read_pom()

        for repo in pom_dict["project"]["repositories"]["repository"]:
            if "http:" in repo["url"]:
                repo["url"] = repo["url"].replace("http:", "https:")
            if repo["id"] == "osgeo":
                repo["url"] = "https://repo.osgeo.org/repository/release/"

        new_repo = {
            "id": "jasperreport-artifactory",
            "url": "https://jaspersoft.jfrog.io/artifactory/third-party-ce-artifacts",
        }
        pom_dict["project"]["repositories"]["repository"].append(new_repo)

        excludes = {
            "excludes": {
                "exclude": [
                    "**/DataStatisticsEventStoreTest.java",
                    "**/DataStatisticsStoreTest.java",
                ]
            }
        }
        for profile in pom_dict["project"]["profiles"]["profile"]:
            if profile["id"] == "default":
                for plugin in profile["build"]["plugins"]["plugin"]:
                    if plugin["artifactId"] == "maven-surefire-plugin":
                        plugin["configuration"].update(excludes)

        write_pom(pom_dict)

    elif bugid in ["Bears-172"]:
        with open(checkout_dir / "molgenis-security/pom.xml") as pom_file:
            pom_dict = xmltodict.parse(
                pom_file.read(), force_list={"exclude", "plugin", "repository"}
            )
            pom_dict = http_repo_to_https(pom_dict)

            new_repo = {
                "id": "cantabria",
                "url": "http://maven.cantabria.es/artifactory/repo/",
            }
            pom_dict["project"]["repositories"]["repository"].append(new_repo)

        with open(checkout_dir / "molgenis-security/pom.xml", "w") as pom_file:
            pom_file.write(xmltodict.unparse(pom_dict, pretty=True))

    elif bugid in ["Bears-179"]:
        pom_dict = read_pom()

        new_plugin_repo = {
            "pluginRepositories": {
                "pluginRepository": {
                    "id": "gradle",
                    "url": "https://repo.gradle.org/gradle/libs-releases-local/",
                }
            }
        }

        pom_dict["project"].update(new_plugin_repo)

        write_pom(pom_dict)

    elif bugid in ["Bears-202"]:
        pom_dict = read_pom()

        plugin_config = {
            "configuration": {"excludes": {"exclude": "**/Issue128Test.java"}}
        }
        for plugin in pom_dict["project"]["build"]["pluginManagement"]["plugins"][
            "plugin"
        ]:
            if plugin["artifactId"] == "maven-surefire-plugin":
                plugin.update(plugin_config)

        write_pom(pom_dict)

    elif bugid in ["Bears-220"]:
        src_path = "src/test/java/com/github/underscore/lodash/LodashTest.java"
        line_numbers = [469, 485, 495, 521, 533]
        ignore_test = "@org.junit.Ignore\n"
        with open(checkout_dir / src_path) as file:
            lines = file.readlines()
        for line_number in line_numbers:
            lines.insert(line_number - 1, ignore_test)
        with open(checkout_dir / src_path, "w") as file:
            file.writelines(lines)

    elif bugid in ["Bears-233"]:
        src_path = "pippo-content-type-parent/pippo-fastjson/src/test/java/ro/pippo/fastjson/FastjsonEngineTest.java"
        ignore_test = "@org.junit.Ignore\n"
        with open(checkout_dir / src_path) as file:
            lines = file.readlines()
        for i, line in enumerate(lines[:]):
            if "public void testDates()" in line:
                lines.insert(i - 1, ignore_test)
        with open(checkout_dir / src_path, "w") as file:
            file.writelines(lines)

    elif bugid in ["Bears-239"]:
        pom_dict = read_pom()

        content = [
            "**/Test262.java",
            "**/EarlyTest.java",
            "**/FailTest.java",
            "**/PassTest.java",
        ]

        for plugin in pom_dict["project"]["build"]["plugins"]["plugin"]:
            if plugin["artifactId"] == "maven-surefire-plugin":
                plugin["configuration"]["excludes"]["exclude"].extend(content)

        write_pom(pom_dict)


@contextlib.contextmanager
def working_environment(bugid: str, branch: str, source_path: str):
    # Checkout the buggy version
    checkout_dir = bears_tmp_dir / f"{bugid}"
    checkout_source(branch, True, checkout_dir)

    target_file_path = checkout_dir / source_path

    source_file_path = bears_tmp_dir / "sources" / bugid / source_path
    source_file_path.parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(target_file_path, source_file_path)

    try:
        yield checkout_dir, source_file_path, target_file_path
    finally:
        shutil.rmtree(checkout_dir)
        shutil.rmtree(bears_tmp_dir / "sources" / bugid)


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
    """Check Java installation"""

    try:
        java_version_string = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, check=True
        ).stderr
    except subprocess.CalledProcessError as e:
        print("Can't find `java`")

    java_pattern = r'"(\d+\.\d+).*"'
    java_version = re.search(java_pattern, java_version_string).groups()[0]

    # RepairThemAll uses Java 8 for Bears
    # https://github.com/program-repair/RepairThemAll/blob/master/script/core/benchmarks/Bears.py
    # This Java version can be installed using SDKMAN (TravaOpenJDK)
    assert java_version == "1.8", "Wrong Java version, needs Java 1.8"

    # Check Maven installation
    try:
        maven_version_string = subprocess.run(
            ["mvn", "-version"], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        print("Can't find `mvn`")

    maven_pattern = r" (\d+\.\d+)\.\d+ "
    maven_version = re.search(maven_pattern, maven_version_string).groups()[0]

    assert maven_version == "3.6", "Wrong Maven version, needs Maven 3.6"


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
    with open(source_file_path, encoding="cp1256") as file:
        lines = file.readlines()
    if bug_len == 0:
        lines.insert(bug_line, indent + patch + "\n")
    else:
        lines[bug_line - 1 : (bug_line - 1) + bug_len] = indent + patch + "\n"

    with open(target_file_path, "w", encoding="cp1256") as file:
        file.writelines(lines)


class Status(Enum):
    PLAUSIBLE = auto()
    COMPILABLE = auto()
    TIMEOUT = auto()
    UNCOMPILABLE = auto()


def run_tests(
    bugid: str, project_dir: Path, pom_path: str, trigger_tests: list[str]
) -> Status:
    single_timeout = 300  # seconds
    total_timeout = 900  # seconds

    mvn_args = "-B -Denforcer.skip=true -Dskip.npm=true -Dskip.gulp=true -Dskip.js=true -Dskip.installnodenpm -Dcheckstyle.skip=true"

    if bugid in ["Bears-149"]:
        java_home = str(java8_home_trava)
    else:
        java_home = str(java8_home_tem)
    env = {**os.environ, "JAVA_HOME": java_home}
    working_dir = (project_dir / pom_path).parent
    compile_cmd = f"cd {working_dir}; mvn compile -DskipTests {mvn_args}"

    if bugid in ["Bears-157", "Bears-172", "Bears-179", "Bears-233"]:
        compile_cmd = f"cd {working_dir}; mvn install -DskipTests {mvn_args}"

    compile_result = run_cmd(compile_cmd, check=False, env=env)
    if compile_result.returncode != 0:
        return Status.UNCOMPILABLE

    # Run triggering tests
    for trigger_test in trigger_tests:
        cmd = f"cd {working_dir}; mvn test -Dtest={trigger_test} -am -DfailIfNoTests=false {mvn_args}"
        result = run_cmd(cmd, timeout=single_timeout, check=False, env=env)

        if result.returncode == 124:
            return Status.TIMEOUT
        elif result.returncode != 0:
            return Status.COMPILABLE

    # Don't run all the tests
    if bugid in ["Bears-179"]:
        return Status.PLAUSIBLE

    # Run the whole test suite
    test_cmd = f"cd {working_dir}; mvn test {mvn_args}"

    result = run_cmd(test_cmd, timeout=total_timeout, check=False, env=env)
    if result.returncode == 124:
        return Status.TIMEOUT
    elif result.returncode != 0:
        return Status.COMPILABLE

    return Status.PLAUSIBLE


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
        return

    mdi = int(bugid.split("-")[1]) - 1

    if len(hunks) == 1 or bugid in ["Bears-8", "Bears-122"]:
        if bugid in ["Bears-8", "Bears-122"]:
            hunk = hunks[0]
            bug_hunk_subset_df = get_hunk_candidates(cp_df, 0)
            cp_df = bug_hunk_subset_df.copy()

        proj_pom_path = bears_bugs[mdi]["reproductionBuggyBuild"]["projectRootPomPath"]
        buggy_buid_id = bears_bugs[mdi]["builds"]["buggyBuild"]["id"]
        pom_path = proj_pom_path.split(str(buggy_buid_id) + "/")[-1]

        assert bears_bugs[mdi]["bugId"] == bugid, f"{bears_bugs[mdi]['bugId']} {bugid}"
        branch = bears_bugs[mdi]["bugName"]
        trigger_tests = [
            t["testClass"] for t in bears_bugs[mdi]["tests"]["failingClasses"]
        ]

        hunk = hunks[0]
        bug_hunk_subset_df = get_hunk_candidates(cp_df, 0)
        bug_line, bug_len = hunk["removed_line_numbers_range"]

        indent_size = len(hunk["added_lines"]) - len(hunk["added_lines"].lstrip(" \t"))
        indent = hunk["added_lines"][:indent_size]

        with working_environment(bugid, branch, hunk["source_path"]) as (
            checkout_dir,
            source_file_path,
            target_file_path,
        ):
            process_projects(bugid, checkout_dir, pom_path)

            for index, patch in bug_hunk_subset_df["decoded_sequences"].items():
                insert_patch(
                    patch, source_file_path, target_file_path, bug_line, bug_len, indent
                )

                # Call the testing infrastructure
                start_timer = timeit.default_timer()
                passed = run_tests(bugid, checkout_dir, pom_path, trigger_tests)
                end_timer = timeit.default_timer()
                cp_df.at[index, "validation_time"] = end_timer - start_timer

                if passed is Status.PLAUSIBLE:
                    cp_df.at[index, "plausible"] = True
                    cp_df.at[index, "compilable"] = True
                elif passed is Status.COMPILABLE:
                    cp_df.at[index, "compilable"] = True
                elif passed is Status.TIMEOUT:
                    cp_df.at[index, "timeout"] = True
                    cp_df.at[index, "compilable"] = True

        cp_df.to_json(save_state_dir / f"{bugid}.jsonl", orient="records", lines=True)

    elif bugid in ["Bears-5", "Bears-106", "Bears-151", "Bears-213"]:
        merged_df = cp_df.loc[
            cp_df["hunk"] == 0,
            ["sequences_scores", "decoded_sequences", "normalized_patch"],
        ].merge(
            cp_df.loc[
                cp_df["hunk"] == 1,
                ["sequences_scores", "decoded_sequences", "normalized_patch"],
            ],
            how="inner",
            on="normalized_patch",
        )
        if len(hunks) > 2:
            for i in range(2, len(hunks)):
                merged_df = merged_df.merge(
                    cp_df.loc[
                        cp_df["hunk"] == i,
                        ["sequences_scores", "decoded_sequences", "normalized_patch"],
                    ],
                    on="normalized_patch",
                    suffixes=(f"_x{i}", f"_y{i}"),
                )

        merged_df["merged_score"] = merged_df.sum(axis=1, numeric_only=True)
        new_cp_df = cp_df.loc[
            (cp_df["hunk"] == 1)
            & (cp_df["normalized_patch"].isin(merged_df["normalized_patch"]))
        ].copy()
        for patch, score in merged_df[["normalized_patch", "merged_score"]].values:
            new_cp_df.loc[
                new_cp_df["normalized_patch"] == patch, "sequences_scores"
            ] = score

        new_cp_df.sort_values(by="sequences_scores", ascending=False, inplace=True)

        ##################################################################

        proj_pom_path = bears_bugs[mdi]["reproductionBuggyBuild"]["projectRootPomPath"]
        buggy_buid_id = bears_bugs[mdi]["builds"]["buggyBuild"]["id"]
        pom_path = proj_pom_path.split(str(buggy_buid_id) + "/")[-1]

        # print(bears_bugs[i]["bugId"])
        assert bears_bugs[mdi]["bugId"] == bugid, f"{bears_bugs[mdi]['bugId']} {bugid}"
        branch = bears_bugs[mdi]["bugName"]
        trigger_tests = [
            t["testClass"] for t in bears_bugs[mdi]["tests"]["failingClasses"]
        ]

        # Checkout the buggy version
        checkout_dir = bears_tmp_dir / f"{bugid}"
        checkout_source(branch, True, checkout_dir)

        # TODO: I can modify `working_environment` to take a list of hunks and return a list of
        # source and target paths. Then I can unify the single and multi-hunk handling codes
        # and use the for loop to process both.
        for hunk in hunks:
            source_path = hunk["source_path"]
            target_file_path = checkout_dir / source_path
            source_file_path = bears_tmp_dir / "sources" / bugid / source_path
            source_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target_file_path, source_file_path)

        for index, patch in new_cp_df["decoded_sequences"].items():
            bugs_lens = defaultdict(list)

            for hunk in hunks:
                source_path = hunk["source_path"]
                target_file_path = checkout_dir / source_path
                source_file_path = bears_tmp_dir / "sources" / bugid / source_path

                bug_line, bug_len = hunk["removed_line_numbers_range"]

                indent_size = len(hunk["added_lines"]) - len(
                    hunk["added_lines"].lstrip(" \t")
                )
                indent = hunk["added_lines"][:indent_size]

                insert_patch(
                    patch,
                    target_file_path
                    if target_file_path in bugs_lens
                    else source_file_path,
                    target_file_path,
                    bug_line + len(bugs_lens[target_file_path])
                    if bug_len == 0
                    else bug_line
                    - (
                        sum(bugs_lens[target_file_path])
                        - len(bugs_lens[target_file_path])
                    ),
                    bug_len,
                    indent,
                )

                bugs_lens[target_file_path].append(bug_len)

            # call the testing infrastructure
            start_timer = timeit.default_timer()
            passed = run_tests(bugid, checkout_dir, pom_path, trigger_tests)
            end_timer = timeit.default_timer()
            new_cp_df.at[index, "validation_time"] = end_timer - start_timer

            if passed is Status.PLAUSIBLE:
                new_cp_df.at[index, "plausible"] = True
                new_cp_df.at[index, "compilable"] = True
            elif passed is Status.COMPILABLE:
                new_cp_df.at[index, "compilable"] = True
            elif passed is Status.TIMEOUT:
                new_cp_df.at[index, "timeout"] = True
                new_cp_df.at[index, "compilable"] = True

        shutil.rmtree(checkout_dir)
        shutil.rmtree(bears_tmp_dir / "sources" / bugid)

        if not new_cp_df.empty:
            new_cp_df.to_json(
                save_state_dir / f"{bugid}.jsonl", orient="records", lines=True
            )

        ##################################################################


def copy_dataset_files(dataset_dir, temp_dataset_dir):
    shutil.copytree(dataset_dir, temp_dataset_dir, dirs_exist_ok=True)


def get_file_path(dir_path: Path, class_name: str) -> Path:
    return dir_path / f"{class_name.replace('.', '/')}.java"


def run_cmd(
    cmd: str,
    check: bool = True,
    timeout: Optional[int] = None,
    env: Optional[Mapping] = None,
) -> subprocess.CompletedProcess[str]:
    args = cmd
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
            env=env,
            shell=True,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124)
    return result


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
    run_cmd(f"git -C {bears_root} worktree prune")
    run_cmd(f"git -C {checkout_dir} init")
    run_cmd(f'git -C {checkout_dir} commit -m "init" --allow-empty')


def main():
    n_jobs = 2
    mp.set_start_method("spawn")

    check_java_version()

    bears_tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(bears_tmp_dir / "sources", ignore_errors=True)
    for directory in bears_tmp_dir.iterdir():
        if directory.name.startswith("Bears-"):
            shutil.rmtree(directory)
    run_cmd(f"git -C {bears_root} worktree prune")

    with open(gen_dir / bugs_metadata_file) as meta_file:
        bugs_metadata = ChainMap(*[json.loads(line) for line in meta_file][::-1])

    candidate_patches_df = pd.read_json(
        output_dir / f"final_candidates_{output_size}.jsonl",
        orient="records",
        lines=True,
    )
    candidate_patches_df["plausible"] = False
    candidate_patches_df["compilable"] = False
    candidate_patches_df["timeout"] = False
    candidate_patches_df["validation_time"] = np.nan

    save_state_dir.mkdir(exist_ok=True)

    # pandas DataFrame is neither thread-safe for reading nor for writing
    # https://stackoverflow.com/questions/13592618/python-pandas-dataframe-thread-safe

    with tqdm_joblib(tqdm(total=len(bugs_metadata))):
        Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(apply_patch)(
                deepcopy(get_candidates(candidate_patches_df, bugid)), bugid, hunks
            )
            for bugid, hunks in bugs_metadata.items()
        )

    cp_dfs = [
        pd.read_json(cp, orient="records", lines=True)
        for cp in save_state_dir.iterdir()
    ]
    concatenated_cp_df = pd.concat(cp_dfs)

    concatenated_cp_df.to_json(
        output_dir / f"plausible_candidates_{output_size}.jsonl",
        orient="records",
        lines=True,
    )

    bugs_with_plausible_patch = (
        concatenated_cp_df.groupby(["bugid", "hunk"])["plausible"]
        .any()
        .groupby("bugid")
        .all()
    )
    print(bugs_with_plausible_patch)
    print(bugs_with_plausible_patch.value_counts())


if __name__ == "__main__":
    main()
