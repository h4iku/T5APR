from pathlib import Path

project_root: Path = Path(__file__).parent.parent
datasets_root: Path = project_root / "datasets"
benchmarks_root: Path = project_root / "benchmarks"
generated_assets_root: Path = project_root / "generated_assets"

cache_dir: Path = project_root / ".cache"

models_root: Path = generated_assets_root / "models"
cloned_models_dir: Path = generated_assets_root / "cloned_models"
