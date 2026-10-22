"""Reproduce delayed-clock time-limit classification without any ML fitting."""
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

from component_inspector.vision.resources import run_bounded


def main():
    root=Path('components/outputs/recall-warm-small'); output=root.parent/'recall-warm-small-recovery-audit'
    output.mkdir(exist_ok=False)
    failed=json.loads((root/'dev-f2-s4217-longer108-resources.json').read_text())
    assert failed['status']=='time_limit' and failed['elapsed_seconds']>failed['time_cap_seconds']
    assert failed['peak_sampled_rss_bytes']<failed['rss_cap_bytes']
    phase=root/'dev-f2-s4217-longer108'
    assert (phase/'plan.json').exists() and not (phase/'model.pt').exists() and not (phase/'training.json').exists()
    preserved={str(path):{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
               for path in [root/'dev-f2-s4217-longer108-resources.json',root/'dev-f2-s4217-longer108-worker.log',phase/'plan.json']}
    command=[sys.executable,'-c','import time; time.sleep(60)']
    control=run_bounded(command,output/'control.log',max_rss_bytes=1024**3,max_seconds=.1)
    assert control['status']=='time_limit' and control['elapsed_seconds']<5
    calls=0
    def jumped_clock():
        nonlocal calls
        calls+=1; return 0. if calls==1 else failed['elapsed_seconds']
    started=time.perf_counter()
    with patch('component_inspector.vision.resources.time.monotonic',side_effect=jumped_clock):
        jump=run_bounded(command,output/'clock-jump.log',max_rss_bytes=1024**3,max_seconds=1800)
    actual=time.perf_counter()-started
    assert jump['status']=='time_limit' and jump['elapsed_seconds']==failed['elapsed_seconds'] and actual<5
    result={'actual_failure':failed,'preserved_failure_files':preserved,'no_failed_checkpoint_exists':True,
            'control':control,'simulated_clock_jump':jump,'clock_jump_actual_harness_seconds':actual,
            'interpretation':'A delayed/jumping monotonic clock reproduces the time-limit status and elapsed overrun without any training. This does not by itself identify whether the original cause was sleep, scheduling, or resource pressure.',
            'no_ml_fit_or_official_test_access':True}
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
