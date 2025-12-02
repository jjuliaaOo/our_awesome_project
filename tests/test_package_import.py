"""
Tests for the promoter_classifier package.
"""

def test_import_and_version():
    """Test that the package can be imported and has a valid version."""
    import promoter_classifier

    assert hasattr(promoter_classifier, "__version__")
    assert isinstance(promoter_classifier.__version__, str)
    assert promoter_classifier.__version__


def test_get_project_info():
    """Test that get_project_info returns the expected structure."""
    import promoter_classifier

    info = promoter_classifier.get_project_info()
    for key in ("title", "description", "version"):
        assert key in info
        assert isinstance(info[key], str)
        assert info[key]
        
    # Also check the additional fields we added
    for key in ("author", "author_email"):
        assert key in info
        assert isinstance(info[key], str)
        assert info[key]