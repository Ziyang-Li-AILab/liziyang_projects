"""ACE-Ego-Hand reproduction: the training / data / eval stack the release omits.

Everything here layers on top of the untouched official inference code in
``../code`` (imported as ``ace_ego_hand``). See ../inventory.md for what was
missing and ../gaps_filled.md for where every value comes from.
"""
import os
import sys

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPRO_ROOT = os.path.dirname(SRC_ROOT)
CODE_ROOT = os.path.join(REPRO_ROOT, "code")

for _p in (CODE_ROOT, os.path.join(CODE_ROOT, "third_party")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
