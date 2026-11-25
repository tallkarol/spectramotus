#!/usr/bin/env python3
"""Package an approved portrait with the offline standard-library Pi launcher."""
import argparse
from pathlib import Path
import shutil
import zipfile
import hashlib
import json
import subprocess
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('pack',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copyfile(args.pack,out/'portrait.zip')
for name in ('launch.py','camera_bridge.py','doctor.py','setup.sh','README.md','TEST-DRIVE.md'):shutil.copyfile(root/'pi'/name,out/name)
(out/'player').mkdir();shutil.copyfile(root/'spectramotus/studio/static/preview.html',out/'player/preview.html')
shutil.copytree(root/'spectramotus/studio/static/runtime',out/'player/runtime')
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
(out/'release.json').write_text(json.dumps({'source_commit':revision,'built_at':datetime.now(timezone.utc).isoformat(),'portrait_sha256':hashlib.sha256((out/'portrait.zip').read_bytes()).hexdigest(),'target':'Raspberry Pi 5 4 GB, Raspberry Pi OS Desktop 64-bit','hardware_verified':False},indent=2)+'\n')
files=sorted(p for p in out.rglob('*') if p.is_file())
(out/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(out))+'\n' for f in files))
archive=out.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(out.rglob('*')):
  if f.is_file():z.write(f,str(Path(out.name)/f.relative_to(out)))
print(archive)
