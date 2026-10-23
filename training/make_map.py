"""A synthetic flat for the training simulator (not anyone's real home): 10 x 7 m, four rooms, doorways."""
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
res = .05
w, h = 200, 140
img = np.full((h, w), 254, np.uint8)
def wall(x0, y0, x1, y1):            # metres, axis-aligned, 10 cm thick
    c0, c1 = int(min(x0, x1) / res), int(max(x0, x1) / res) + 2
    r0, r1 = int(min(y0, y1) / res), int(max(y0, y1) / res) + 2
    img[h - r1:h - r0, c0:c1] = 0
wall(0, 0, 9.9, 0); wall(0, 6.9, 9.9, 6.9); wall(0, 0, 0, 6.9); wall(9.9, 0, 9.9, 6.9)
wall(4.0, 0, 4.0, 2.4); wall(4.0, 3.3, 4.0, 6.9)              # corridor wall with a doorway
wall(0, 3.5, 1.6, 3.5); wall(2.5, 3.5, 4.0, 3.5)              # left rooms split, doorway 0.9 m
wall(4.0, 4.2, 6.5, 4.2); wall(7.4, 4.2, 9.9, 4.2)            # right rooms split, doorway 0.9 m
for x, y, sx, sy in ((1.0, 5.5, .8, .5), (6.0, 1.0, 1.2, .6), (8.6, 5.8, .6, .6), (2.8, 1.2, .5, .5)):
    wall(x, y, x + sx, y)                                     # furniture blocks
    img[h - int((y + sy) / res):h - int(y / res), int(x / res):int((x + sx) / res)] = 0
(out / 'map.pgm').write_bytes(b'P5\n%d %d\n255\n' % (w, h) + img.tobytes())
(out / 'map.yaml').write_text('image: map.pgm\nmode: trinary\nresolution: 0.050000\norigin: [0.000000, 0.000000, 0.0]\n'
                              'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n', encoding='utf-8')
