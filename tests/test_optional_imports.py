"""Base import smoke tests for optional dependency guarding."""


def test_base_package_imports_without_optional_extras():
    import nucleisky2d  # noqa: F401
    import nucleisky3d  # noqa: F401
