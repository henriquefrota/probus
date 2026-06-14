"""Shared helpers for rule implementations."""


def is_test_filepath(filepath: str) -> bool:
    """Return True if the path points at a test file or lives under a tests/ tree.

    Test files exercise the rules against intentionally broken fixtures, so the
    rules must not flag them as production code. Matches paths containing
    '/tests/' or '/test_', or basenames beginning with 'test_'.
    """
    normalized = filepath.replace("\\", "/")
    return (
        "/tests/" in normalized
        or "/test_" in normalized
        or normalized.startswith("test_")
    )
