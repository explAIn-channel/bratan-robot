"""The author's apartment map is not published. Tests that drive on a real map run only when one is put in tests/data
(apartment-final-20261006-dock.pgm/.yaml, rooms.json, rooms-labels.npy, scans-1010-doorway.json)."""
from pathlib import Path

DATA = Path(__file__).parent / 'data'
HAVE_MAP = (DATA / 'apartment-final-20261006-dock.yaml').exists()

NEEDS_MAP = [
    'test_checklist.py', 'test_day1010.py', 'test_doors.py', 'test_ears.py', 'test_growbot.py',
    'test_localize_clutter.py', 'test_mind.py', 'test_nullvoice.py', 'test_passages.py', 'test_recognize.py',
    'test_scenarios.py', 'test_senses.py', 'test_sound_map.py', 'test_talk_actions.py', 'test_track.py',
]
MAP_TESTS = {'test_match_recovers_offset_quickly', 'test_global_recovery', 'test_plan_between_rooms',
             'test_dock_reference_consensus'}

collect_ignore = [] if HAVE_MAP else NEEDS_MAP


def pytest_collection_modifyitems(config, items):
    if HAVE_MAP:
        return
    import pytest
    skip = pytest.mark.skip(reason='needs the apartment map in tests/data')
    for item in items:
        if item.name in MAP_TESTS:
            item.add_marker(skip)
