#!/usr/bin/env python3
"""Package an approved portrait with the same builder used by admin downloads."""
import argparse
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from spectramotus.studio.pi_kit import build_kit
p=argparse.ArgumentParser();p.add_argument('pack',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
print(build_kit(args.pack,args.output.resolve(),revision=revision))
