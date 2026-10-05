"""Compatibility CLI for the native library's evidence store."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agxforge.g17.evidence import *

if __name__ == "__main__":
    # sys.exit(main()), not main(): the return value IS the verdict. `status` returned 1 for an
    # unextracted tree and this line threw it away, so the check reported "extracted": false and
    # exited 0 - a status command that could not fail, inside the change that added it to catch
    # exactly that.
    sys.exit(main())
