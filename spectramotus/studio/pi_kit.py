"""Admin-only offline kit assembled from private demo media and shipped player code."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import zipfile
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from .navigation import render_page
from .packs import STATIC, digest, make_pack, normalized_photo

PI_FILES=('launch.py','camera_bridge.py','doctor.py','setup.sh','README.md','TEST-DRIVE.md')
PI_ROOT=Path(__file__).resolve().parents[2]/'pi'
CLIPS=(('wave','Wave','reaction'),('thumbs_up','Thumbs-up','reaction'),('idle_blink','Blink','idle'))


def build_kit(pack, out, pi_root=PI_ROOT, revision=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(pack,out/'portrait.zip')
    for name in PI_FILES:shutil.copyfile(pi_root/name,out/name)
    (out/'player').mkdir();shutil.copyfile(STATIC/'preview.html',out/'player/preview.html')
    shutil.copytree(STATIC/'runtime',out/'player/runtime')
    (out/'release.json').write_text(json.dumps({'source_commit':revision,'built_at':datetime.now(timezone.utc).isoformat(),'portrait_sha256':digest(out/'portrait.zip'),'target':'Raspberry Pi 5 4 GB, Raspberry Pi OS Desktop 64-bit','hardware_verified':False},indent=2)+'\n')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    (out/'SHA256SUMS').write_text(''.join(digest(f)+'  '+str(f.relative_to(out))+'\n' for f in files))
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,str(Path(out.name)/f.relative_to(out)))
    return archive


def mount_pi_kit(app, root, demo):
    cache=Path(root)/'pi-kits';demo=Path(demo);lock=threading.Lock()
    def sources():
        required=[demo/'photo.jpg',*[demo/(ident+'.mp4') for ident,_,_ in CLIPS],*[PI_ROOT/name for name in PI_FILES],STATIC/'preview.html']
        required+=sorted(p for p in (STATIC/'runtime').rglob('*') if p.is_file())
        if not all(p.is_file() for p in required):raise HTTPException(409,'The current Karol kit is not available yet. Please contact the studio owner.')
        if (demo/'clip-timing.json').is_file():required.append(demo/'clip-timing.json')
        # Include the packaging code so cache invalidation covers all release inputs.
        required += [Path(__file__),Path(__file__).with_name('packs.py')]
        return required
    def archive():
        with lock:
            files=sources();key=hashlib.sha256(''.join(digest(p) for p in files).encode()).hexdigest()
            cache.mkdir(parents=True,exist_ok=True);dest=cache/(key+'.zip')
            if dest.exists():return dest
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                folder=Path(tmp)/'content/pack';(folder/'clips').mkdir(parents=True)
                normalized_photo(demo/'photo.jpg').save(folder/'rest.png')
                timing=json.loads((demo/'clip-timing.json').read_text()) if (demo/'clip-timing.json').exists() else {}
                clips=[]
                for ident,label,role in CLIPS:
                    path=demo/(ident+'.mp4');shutil.copyfile(path,folder/'clips'/path.name)
                    duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)],text=True,timeout=15))
                    clip={'id':ident,'label':label,'file':'clips/'+path.name,'kind':'existing','role':role,'duration_seconds':duration,'starts_at':'rest','ends_at':'rest'}
                    for field in ('motion_onset_seconds','preparation'):
                        if field in timing.get(ident,{}):clip[field]=timing[ident][field]
                    clips.append(clip)
                manifest={'schema_version':1,'pack_id':'karol-pi-'+key[:12],'name':'Karol · Pi 5 test drive','version':1,'minimum_player_version':'0.1','canvas':{'width':832,'height':1104,'fps':24},'rest':'rest.png','credits_spent':0,'mode':'existing','clips':clips,'events':{'wave':'wave','thumbs_up':'thumbs_up'},'policy':{'interrupt':False,'cooldown_ms':0,'idle':{'clip_id':'idle_blink','delay_ms':4000}}}
                pack=make_pack(folder,manifest)
                release=build_kit(pack,Path(tmp)/'Spectramotus-Pi5-Karol',revision=os.getenv('RAILWAY_GIT_COMMIT_SHA'))
                release.replace(dest)
            return dest
    @app.get('/admin/pi',response_class=HTMLResponse)
    def page(request:Request):return render_page(request,'pi-kit.html')
    @app.get('/api/admin/pi-kit')
    def info():
        try:sources();available=True
        except HTTPException:available=False
        return {'available':available,'hardware_verified':False,'generation_enabled':False}
    @app.get('/api/admin/pi-kit/download')
    def download():
        path=archive()
        return FileResponse(path,media_type='application/zip',filename='Spectramotus-Pi5-Karol.zip',headers={'X-Kit-SHA256':digest(path)})
