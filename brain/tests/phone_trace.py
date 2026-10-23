import json, sys, time
sys.path.insert(0, '/opt/robotbrain/bratan/app')
from bratan.common import bus
last = [None]
def on(topic, d):
    if topic == 'body.state':
        sk = d.get('skill') or {}
        line = (sk.get('stage'), sk.get('sub'), sk.get('door_phase'), d.get('blocked'))
        print(time.strftime('%H:%M:%S'), [round(v, 2) for v in d['pose']], 'tw', [round(v, 2) for v in d['twist']], 'int', d.get('intent'), line, flush=True)
    elif topic == 'body.event':
        print('EVENT', d, flush=True)
c = bus.Client('trace', ('body.state', 'body.event'), on).start()
c.wait_connected(3)
c.publish('body.do', json.loads(sys.argv[1]))
time.sleep(float(sys.argv[2]))
c.publish('body.stop', {})
time.sleep(.5)
