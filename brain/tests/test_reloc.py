"""Relocalize when wedged, with a fake body (no robot): python -m pytest tests/test_reloc.py

10.10 11:13: lost with the nose against a panel by the speaker — «тесно — развернуться не дало препятствие», six
views without a turn, lost for two hours while the battery ran down.
"""
from bratan.body import skills


class WedgedCtx:
    """Lost, nose to a panel: no turn is clear; backing is, until a wall half a metre behind."""

    def __init__(self):
        self.odom, self.now, self.recovering, self.localized, self.recoveries = (0., 0., 0.), 0., False, False, 0

    def clear(self, v, w, seconds):
        return v < 0 and w == 0 and self.odom[0] > -.5

    def request_recovery(self):
        self.recoveries += 1


def run(ctx, found_after_m=None):
    skill = skills.Relocalize({})
    intents = []
    for _ in range(400):
        intent = skill.tick(ctx)
        intents.append(intent)
        if skill.status:
            break
        ctx.now += .1
        ctx.odom = (ctx.odom[0] + intent.v * .1, 0., 0.)
        if found_after_m is not None and ctx.odom[0] <= -found_after_m:
            ctx.localized = True
    return skill, intents


def test_lost_and_wedged_backs_out_before_looking_again():
    ctx = WedgedCtx()
    skill, intents = run(ctx)
    assert any(i.v < 0 for i in intents)                               # it backed out
    assert -.65 < ctx.odom[0] < -.2                                   # a bounded step back, the wall respected
    assert skill.status == 'failed' and skill.reason == 'lost'        # still lost here: it says so in the end


def test_backing_out_finds_the_place():
    ctx = WedgedCtx()
    skill, _ = run(ctx, found_after_m=.2)
    assert skill.status == 'done'
