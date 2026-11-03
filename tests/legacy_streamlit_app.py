"""Internal compatibility harness for the pre-unification UI regression tests."""

from pathlib import Path
from runpy import run_path

app = Path(__file__).resolve().parents[1] / "streamlit_app.py"
namespace = run_path(str(app), run_name="legacy_streamlit_harness")
namespace["legacy_main"]()
