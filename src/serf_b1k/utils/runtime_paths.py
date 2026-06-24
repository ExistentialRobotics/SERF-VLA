import sys
from pathlib import Path


def ensure_repo_src_on_path(start: str | Path | None = None) -> Path:
    current = Path(start or __file__).resolve()
    for parent in [current, *current.parents]:
        src_root = parent / "src"
        if (src_root / "serf_b1k" / "__init__.py").exists():
            src_root_str = str(src_root)
            if src_root_str not in sys.path:
                sys.path.insert(0, src_root_str)
            return src_root
    raise ImportError("Could not find repository src/ directory containing serf_b1k.")
