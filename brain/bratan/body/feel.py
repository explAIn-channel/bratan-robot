"""Feeling hands on the body (GrowBot, 11.10: the model read the IMU and said «меня покачали, подняли»).

The board gives the gyro rate (gyro_w) and the wheels' odometry. When I am not asking the wheels for anything and
the body still turns or moves, someone did it: turned by hand (the wheels did not turn with it — most likely
lifted) or pushed along the floor. My own motion has a tail after a stop, so the first QUIET_S after any command
does not count. On the dock nothing is felt: pressing into the contacts moves the body a little by itself.
"""
from __future__ import annotations

import math

QUIET_S = .8            # after my own last command: the wheels and the body settle
TURN_RATE = .5          # rad/s of the gyro that is a hand, not noise (resting noise is ~0.01)
TURN_RAD = .5           # ~30 degrees turned by someone else
PUSH_M = .05            # odometry moved this far without a command
WINDOW_S = 3.           # a push or a turn is counted within this long
COOLDOWN_S = 6.         # one feeling, not a stream of them


class Feel:
    def __init__(self):
        self.quiet_since = None
        self.reset(None, 0.)
        self.told_at = -1e9

    def reset(self, odom, now):
        self.start_odom = tuple(odom[:3]) if odom else None
        self.started = now
        self.turned = 0.
        self.wheels_turned = 0.
        self.last_t = None

    def update(self, body, commanded, docked, now):
        """One board sample. Returns None or {'how': 'turned'|'pushed', ...} once per feeling."""
        odom = body.get('odom')
        gyro = body.get('gyro_w')
        sent = body.get('sent') or (0., 0.)
        if commanded or docked or abs(sent[0]) > 1e-3 or abs(sent[1]) > 1e-3 or odom is None:
            self.quiet_since = None
            return None
        if self.quiet_since is None:
            self.quiet_since = now
        if now - self.quiet_since < QUIET_S or self.start_odom is None or now - self.started > WINDOW_S:
            self.reset(odom, now)
            return None
        dt = 0. if self.last_t is None else max(0., min(.2, now - self.last_t))
        self.last_t = now
        if gyro is not None and abs(gyro) > TURN_RATE:
            self.turned += gyro * dt
        encoder_w = body.get('encoder_w')
        if encoder_w is not None:
            self.wheels_turned += abs(encoder_w) * dt
        if now - self.told_at < COOLDOWN_S:
            return None
        dx, dy = odom[0] - self.start_odom[0], odom[1] - self.start_odom[1]
        moved = math.hypot(dx, dy)
        if abs(self.turned) >= TURN_RAD:
            lifted = encoder_w is not None and self.wheels_turned < .3 * abs(self.turned)
            feeling = {'how': 'turned', 'degrees': round(math.degrees(self.turned)), 'lifted': lifted}
        elif moved >= PUSH_M:
            along = dx * math.cos(self.start_odom[2]) + dy * math.sin(self.start_odom[2])
            feeling = {'how': 'pushed', 'meters': round(moved, 2), 'back': along < 0}
        else:
            return None
        self.told_at = now
        self.reset(odom, now)
        return feeling


def feeling_text(feeling):
    """The feeling in words for the mind."""
    if feeling.get('how') == 'turned':
        side = 'налево' if feeling.get('degrees', 0) > 0 else 'направо'
        text = f'меня повернули руками {side}, градусов на {abs(int(feeling.get("degrees", 0)))}'
        return text + (' — колёса не крутились, наверное, подняли' if feeling.get('lifted') else '')
    if feeling.get('how') == 'pushed':
        # Not «руками»: Taffa bumping into me pushes just the same.
        return (f'меня {"оттолкнули назад" if feeling.get("back") else "подтолкнули вперёд"} '
                f'сантиметров на {int(round(100 * feeling.get("meters", 0)))}, пока я стоял')
    return 'меня кто-то потрогал'
