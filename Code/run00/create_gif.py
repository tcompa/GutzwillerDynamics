#!/usr/bin/env python

import os
import sys

output_dir = 'Figs'
if len(sys.argv) > 1:
    delay = float(sys.argv[1])
else:
    delay = 100

print('creating animation.gif')
os.system('convert -delay %i -dispose Background +page %s/*.png -loop 0 animation.gif' % (delay, output_dir))

