def test_import():
    import importlib
    import sys
    try:
        import lacun  # type: ignore
    except Exception:
        # When extension not built yet, package may still import
        importlib.import_module('lacun')
