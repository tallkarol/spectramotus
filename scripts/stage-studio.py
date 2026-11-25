#!/usr/bin/env python3
"""Make an explicit allow-list deployment bundle; never upload the whole repository."""
import argparse
from pathlib import Path
import shutil

parser=argparse.ArgumentParser()
parser.add_argument('--photo',required=True,type=Path)
parser.add_argument('--wave',required=True,type=Path)
parser.add_argument('--thumbs-up',type=Path)
parser.add_argument('--idle',type=Path)
parser.add_argument('--output',type=Path,default=Path('work/studio-deploy'))
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=args.output.resolve()
if out.exists():
    raise SystemExit('Choose a new empty output directory to avoid including stale files.')
out.mkdir(parents=True)
for source,target in [('Dockerfile.studio','Dockerfile'),('requirements-studio.txt','requirements-studio.txt'),('spectramotus/__init__.py','spectramotus/__init__.py')]:
    dest=out/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/source,dest)
shutil.copytree(root/'spectramotus/studio',out/'spectramotus/studio',ignore=shutil.ignore_patterns('__pycache__'))
(out/'demo').mkdir()
shutil.copyfile(args.photo,out/'demo/photo.jpg')
shutil.copyfile(args.wave,out/'demo/wave.mp4')
if args.thumbs_up: shutil.copyfile(args.thumbs_up,out/'demo/thumbs_up.mp4')
if args.idle: shutil.copyfile(args.idle,out/'demo/idle_blink.mp4')
timing=args.wave.parent/'clip-timing.json'
if timing.exists(): shutil.copyfile(timing,out/'demo/clip-timing.json')
(out/'railway.json').write_text('{"build":{"builder":"DOCKERFILE","dockerfilePath":"Dockerfile"},"deploy":{"healthcheckPath":"/health","healthcheckTimeout":120,"restartPolicyType":"ON_FAILURE","restartPolicyMaxRetries":3}}\n')
print(out)
