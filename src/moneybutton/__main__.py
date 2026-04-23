"""Entry point so `python -m moneybutton <cmd>` resolves to the click CLI.

The cli module is not yet built (see build order step 26); this stub keeps
`python -m moneybutton` importable during early development so the rest of the
package can be tested without the CLI surface being wired up.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from moneybutton.cli import cli
    except ModuleNotFoundError:
        print(
            "moneybutton.cli is not yet implemented (build order step 26). "
            "The package is still under construction.",
            file=sys.stderr,
        )
        return 2
    cli()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
