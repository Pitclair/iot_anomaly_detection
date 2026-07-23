"""Backward-compatible launcher for the installed ``lm-idnet`` command."""

from lm_idnet.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
