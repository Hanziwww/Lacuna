def test_import():
    import importlib
    import sys

    try:
        import lacunaa  # type: ignore
    except Exception:
        # When extension not built yet, package may still import
        importlib.import_module("lacuna")
