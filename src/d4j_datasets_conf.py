from pathlib import Path

project_root: Path = Path(__file__).parent.parent
datasets_root: Path = project_root / "datasets"

d4j_version = "2.0.0"
# d4j_version = "1.4.0"

d4j_root: Path = project_root / f"defects4j-{d4j_version}"
d4j_bin: Path = d4j_root / "framework/bin/defects4j"
d4j_gen_dir: Path = datasets_root / f"GeneratedDefects4J/defects4j-{d4j_version}"
# d4j_gen_dir: Path = datasets_root / f"GeneratedDefects4J/only-d4j-1.4"
# d4j_gen_dir: Path = datasets_root / f"GeneratedDefects4J/only-d4j-2.0"
d4j_tmp_dir: Path = datasets_root / "tmp"

d4j1_gen_dir: Path = datasets_root / f"GeneratedDefects4J/only-d4j-1.4"
d4j2_gen_dir: Path = datasets_root / f"GeneratedDefects4J/only-d4j-2.0"

bears_root: Path = project_root / f"bears-benchmark"
bears_gen_dir: Path = datasets_root / f"GeneratedBears"
bears_tmp_dir: Path = datasets_root / "tmp"

tree_sitter_lib: Path = project_root / "tools/tree-sitter-lib"
