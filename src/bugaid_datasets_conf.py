from pathlib import Path

project_root: Path = Path(__file__).parent / ".."

bugaid_gen_dir: Path = project_root / "datasets/BugAID_Examples"
bugaid_data_dir: Path = project_root / "datasets/BugAID_Examples/data"

codeflaws_gen_dir: Path = project_root / "datasets/Codeflaws"
codeflaws_data_dir: Path = project_root / "datasets/Codeflaws/codeflaws_data"

manybugs_gen_dir: Path = project_root / "datasets/ManyBugs"
manybugs_data_dir: Path = project_root / "datasets/ManyBugs/scenarios/uncompressed"
