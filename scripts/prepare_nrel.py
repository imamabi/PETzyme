"""Run from the repository root: python -m scripts.prepare_nrel."""
import json
from pathlib import Path
from petzyme.nrel import prepare

def main() -> None:
    """Flow: Receives external CLI invocation; returns None after prepare() validation is printed."""
    print(json.dumps(prepare(Path(__file__).resolve().parents[1])["validation"], indent=2))


if __name__ == "__main__":
    main()
