#!/usr/bin/env python3
"""Write a local hardware report without serial numbers, IPs, images, or credentials."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

def command(args):
    if not shutil.which(args[0]):return 'Not installed'
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=12)
        return (r.stdout+r.stderr).strip()[:6000]
    except (OSError,subprocess.TimeoutExpired):return 'Unavailable or timed out'

def report():
    model=Path('/proc/device-tree/model')
    return {'captured_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'model':model.read_text().strip('\x00') if model.exists() else platform.system(),
            'architecture':platform.machine(),'python':platform.python_version(),
            'desktop':os.getenv('XDG_CURRENT_DESKTOP','unknown'),'session':os.getenv('XDG_SESSION_TYPE','unknown'),
            'chromium':command([shutil.which('chromium') or shutil.which('chromium-browser') or 'chromium','--version']),
            'picamera2_available':importlib.util.find_spec('picamera2') is not None,
            'cameras':command(['rpicam-hello','--list-cameras']),
            'temperature':command(['vcgencmd','measure_temp']),
            'throttling':command(['vcgencmd','get_throttled']),
            'memory':command(['free','-m']), 'disk_free_mb':round(shutil.disk_usage(Path.home()).free/1024**2)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('hardware-report.json'));a=p.parse_args()
    result=report();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('Saved:',a.output.resolve())
