import sys
import os


def pytest_sessionstart(session):
    # Ensure project root is on sys.path so tests can import top-level packages like `data`.
    project_root = os.path.abspath(os.getcwd())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Print current working dir and a trimmed sys.path for debug
    print('\n[conftest] CWD=', os.getcwd())
    print('[conftest] sys.path sample:')
    for p in sys.path[:6]:
        print('  ', p)
