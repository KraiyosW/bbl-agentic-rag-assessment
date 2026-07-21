from pathlib import Path

import pytest


@pytest.fixture
def sample_knowledge_base(tmp_path: Path) -> Path:
    path = tmp_path / "knowledge_base.txt"
    path.write_text(
        """[International Travel]
International trips require manager and department-head approval at least 14 days before departure.

[International Travel]
Travelers must complete a security briefing and confirm corporate travel insurance.

[Travel Expenses]
Travel expense claims and itemized receipts must be submitted within 10 business
days after returning.

[Remote Work]
Remote work is allowed two days per week with manager approval.
""",
        encoding="utf-8",
    )
    return path
