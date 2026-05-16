from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_NAME = "2026-05-16-phase-1-sub-B1-singapore-frequency-analysis"


def test_script_runs_against_fixtures_and_produces_well_formed_report(tmp_path: Path) -> None:
    """The script must complete successfully against sub-A's synthetic fixtures.

    Several B1 fields have no matching column in the fixture data (places.categories
    is a plain string column, buildings.subtype/base.class don't exist, etc.); the
    script must handle this gracefully by writing placeholder sections instead of
    raising. The integration test asserts well-formedness only — top-level sections
    present, file readable, no crash. Per-field PNG count and numeric content are
    intentionally NOT asserted.
    """
    fixture_dir = REPO_ROOT / "tests" / "fixtures" / "overture_mini"
    assert fixture_dir.exists(), f"sub-A fixtures missing at {fixture_dir}"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyse_singapore_frequencies.py",
            "--backend",
            "fixture",
            "--fixture-dir",
            str(fixture_dir),
            "--output-dir",
            str(tmp_path),
            "--rerun-reason",
            "integration-test",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"script failed:\nstdout={result.stdout}\nstderr={result.stderr}"

    report_path = tmp_path / f"{REPORT_NAME}.md"
    plots_dir = tmp_path / f"{REPORT_NAME}_plots"

    # File and dir existence.
    assert report_path.exists(), f"report not written at {report_path}"
    assert plots_dir.exists() and plots_dir.is_dir(), f"plots dir not created at {plots_dir}"

    content = report_path.read_text()

    # HTML-comment header on first lines.
    assert content.startswith("<!--"), "report must begin with the HTML-comment header"
    assert "scripts/analyse_singapore_frequencies.py" in content
    assert "Re-run reason: integration-test" in content

    # Provisional status line.
    assert "Status: provisional" in content
    assert "Sweden" in content

    # All five top-level sections present.
    for header in (
        "## 1. Methodology",
        "## 2. Coverage summary",
        "## 3. Field analyses",
        "## 4. Implications for B2",
        "## 5. Reproducibility",
    ):
        assert header in content, f"missing top-level section: {header}"

    # All nine field section headers present (graceful skips still emit the header).
    for n in range(1, 10):
        assert f"### 3.{n}" in content, f"missing field subsection 3.{n}"

    # Implications enumeration.
    assert "emit `<unknown>`" in content
    assert "drop missing-class features" in content
    assert "infer class from context" in content
