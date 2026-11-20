_IMPORT_ERROR = (
    "bohnasa.visualization requires matplotlib, which is only needed by the "
    "'examples' extra: pip install 'bohnasa[examples]' (source checkouts: "
    "pip install -e '.[examples]')"
)


def _pyplot():
    """Import matplotlib.pyplot lazily, raising a friendly error if absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(_IMPORT_ERROR) from exc
    return plt
