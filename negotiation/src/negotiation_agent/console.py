"""Console output for the launchers that print Korean.

A Windows pipe or redirected file uses the locale code page (cp1252 on GitHub's Windows runners),
which cannot encode Korean, so a print raised UnicodeEncodeError and the run exited with status 1.
An interactive Windows console was never affected: Python writes to it through the Unicode console API.
"""
import sys


def utf8_stdio() -> None:
    """Write stdout and stderr as UTF-8 whatever the locale code page is."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
