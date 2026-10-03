"""Mutation testing: break the navigate code on purpose, one fault at a time, and check the
suite or the harness catches each one.

    python3 test/mutate.py            # every mutation (a few minutes)
    python3 test/mutate.py M1 M4      # just those

A check that cannot fail is worse than none (STORY section 4). Each entry below is a real way
the approach could go wrong - steering by odometry, looking with the arms in the way, trusting a
blob cut off by the arms, never scanning from the side, parking badly, overshooting, routing
through an object, localising off the wall centres, turning too close to what was just stood up,
staging where the camera cannot see. Each is applied to a scratch copy of the source, never to
this tree, and the named test or the harness must then FAIL. Result on 2026-09-28: all caught,
except "M5 accept any park", which changes nothing because the first park is already within a
few millimetres; "M5b" is the version of that fault that matters, and it is caught. The two
image-pipeline rules (the arm band and "behind another object") are mutated in D1/D2 when NumPy
and OpenCV are available.
"""
import os
import shutil
import subprocess
import sys
import tempfile

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MUTS = [
  ('M1 steer by odometry, not the wall fix', 'navigate_behavior/navigate_behavior/task_node.py',
   "        if self._last_fix is None:\n            return tuple(self.odom)",
   "        if True:\n            return tuple(self.odom)", ['harness.py', '1']),
  ('M2 scan with the arms parked', 'navigate_behavior/navigate_behavior/sense_task.py',
   "        self._look_move(label, parked, (LOOK_LEFT, LOOK_RIGHT), 'the look pose')\n        dets",
   "        dets", ['harness.py', '1']),
  ('M3 no arm band on the mast camera', 'navigate_behavior/config/task.yaml',
   "    far_v_max: 330", "    far_v_max: 479", ['test_navigate.py:test_the_look_pose_clears_the_mast_camera_above_far_v_max']),
  ('M4 no vantage points', 'navigate_behavior/navigate_behavior/task_node.py',
   "        cands.sort(key=lambda p: math.dist(p, here))\n        return cands[:limit]",
   "        return []", ['harness.py', '3']),
  ('M5 accept any park', 'navigate_behavior/config/task.yaml',
   "    park_tol: 0.010", "    park_tol: 1.0", ['harness.py', '1']),
  ('M5b never drive the last leg', 'navigate_behavior/navigate_behavior/task_node.py',
   "        if not self._drive_to_park(name):\n            return False\n        for attempt",
   "        self.park_tol = 1.0\n        for attempt", ['harness.py', '1']),
  ('M6 no slow-down in the drive law', 'navigate_behavior/navigate_behavior/navigation.py',
   "    v = max(v_min, min(v_max, k_v * along))", "    v = v_max", ['test_navigate.py:test_the_drive_law_arrives_without_overshooting']),
  ('M7 routes ignore objects', 'navigate_behavior/navigate_behavior/navigation.py',
   "        d = seg_seg(p, q, a, b) - r - BASE_RADIUS", "        d = 9.0", ['test_navigate.py:test_routes_keep_clear_and_go_round']),
  ('M8 the room at the wall centres', 'navigate_behavior/config/task.yaml',
   "    room: [-2.5, 2.5, -2.5, 2.5]", "    room: [-2.55, 2.55, -2.55, 2.55]", ['harness.py', '1']),
  ('M9 no backup after a stand-up', 'navigate_behavior/config/task.yaml',
   "    backup: 0.35", "    backup: 0.05", ['test_navigate.py:test_the_backup_clears_the_object_before_any_turn']),
  ('M10 staging past the workspace view', 'navigate_behavior/navigate_behavior/navigation.py',
   "STAGE_REACH = 0.42           #", "STAGE_REACH = 0.52           #", ['test_navigate.py:test_the_camera_ranges_are_what_the_approach_uses']),
  ('D1 no arm band in detect', 'navigate_behavior/navigate_behavior/detect.py',
   "                bottom = h - 1 - edge if v_max is None else min(h - 1, v_max) - edge",
   "                bottom = h - 1 - edge", ['test_navigate.py:test_a_blob_into_the_arm_band_is_clipped']),
  ('D2 no behind rule in detect', 'navigate_behavior/navigate_behavior/detect.py',
   "        det['clipped'] = bool(det['clipped'] or det['behind'])",
   "        det['clipped'] = bool(det['clipped'])",
   ['test_navigate.py:test_an_object_partly_behind_another_is_refused_and_the_front_one_kept']),
]


def has_cv():
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    which = sys.argv[1:]
    missed = 0
    for name, f, old, new, cmd in MUTS:
        if which and name.split()[0] not in which:
            continue
        if name.startswith('D') and not has_cv():
            print('SKIPPED', name, '(needs NumPy and OpenCV)')
            continue
        tmp = tempfile.mkdtemp(prefix='navmut_')
        try:
            shutil.copytree(SRC, os.path.join(tmp, 'src'),
                            ignore=shutil.ignore_patterns('__pycache__', 'build', 'install', 'log'))
            p = os.path.join(tmp, 'src', f)
            s = open(p).read()
            assert old in s, (name, 'the code this mutation targets has changed - update it')
            open(p, 'w').write(s.replace(old, new, 1))
            beh = os.path.join(tmp, 'src', 'navigate_behavior')
            if cmd[0] == 'harness.py':
                r = subprocess.run([sys.executable, 'test/harness.py'] + cmd[1:], cwd=beh,
                                   capture_output=True, text=True)
                fails = [ln.strip() for ln in r.stdout.splitlines() if '[FAIL]' in ln]
                caught = r.returncode != 0
                detail = fails[0][:110] if fails else (r.stderr.strip().splitlines() or [''])[-1][:110]
            else:
                mod, test = cmd[0].split(':')
                r = subprocess.run([sys.executable, '-c',
                                    f"import sys; sys.path.insert(0, 'test'); sys.path.insert(0, '.'); "
                                    f"import {mod[:-3]} as T; T.{test}()"],
                                   cwd=beh, capture_output=True, text=True)
                caught = r.returncode != 0
                detail = (r.stderr.strip().splitlines() or [''])[-1][:110]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        missed += not caught
        print(('CAUGHT ' if caught else 'MISSED ') + name + ' -> ' + detail, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
