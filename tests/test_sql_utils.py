import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_utils import validate_insert_statement
from file_utils import calculate_md5

def test_validate_insert_statement_accepts_simple_insert():
    statement = "INSERT INTO courses (id, name) VALUES (11, 'Philosophy');"
    assert validate_insert_statement(statement) is True


def test_validate_insert_statement_allows_semicolon_inside_string():
    statement = "INSERT INTO notes (body) VALUES ('Part A; Part B');"
    assert validate_insert_statement(statement) is True


def test_validate_insert_statement_rejects_multiple_statements():
    statement = "INSERT INTO courses (id) VALUES (1); DROP TABLE courses;"
    assert validate_insert_statement(statement) is False


def test_validate_insert_statement_rejects_non_insert():
    statement = "UPDATE courses SET name='Math' WHERE id=1"
    assert validate_insert_statement(statement) is False


def test_calculate_md5(tmp_path: Path):
    file_path = tmp_path / "data.bin"
    data = b"example data"
    file_path.write_bytes(data)

    expected_hash = "5c71dbb287630d65ca93764c34d9aa0d"
    assert calculate_md5(str(file_path)) == expected_hash
