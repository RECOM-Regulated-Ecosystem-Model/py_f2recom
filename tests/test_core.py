"""Basic tests for the core module."""

def test_import():
    """Test that the package can be imported."""
    import py_f2recom
    assert py_f2recom is not None
    assert hasattr(py_f2recom, '__version__')

test_import()