"""Single-owner placeholder studio. No paid provider imports or API calls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
import hmac
import io
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from .motus import MotusLibrary, mount_motus
from .packs import STATIC, MAX_UPLOAD, make_pack, photo_signature, runtime_files


def create_app(data_dir=None, password=None, demo_dir=None):
    root=Path(data_dir or os.getenv('STUDIO_DATA_DIR', 'work/studio'))
    root.mkdir(parents=True, exist_ok=True)
    projects=root/'projects'; projects.mkdir(exist_ok=True)
    demo=Path(demo_dir or os.getenv('STUDIO_DEMO_DIR', 'work/studio-demo'))
    secret=password if password is not None else os.getenv('STUDIO_PASSWORD')
    if not secret:
        raise RuntimeError('Set STUDIO_PASSWORD before starting the studio.')
    secure=os.getenv('STUDIO_SECURE_COOKIE','0')=='1'
    lock=threading.Lock()
    pool=ThreadPoolExecutor(max_workers=1)
    attempts={}
    motus=MotusLibrary(root/'motus')

    def save(p, value):
        tmp=p/'project.tmp'; tmp.write_text(json.dumps(value,indent=2)); tmp.replace(p/'project.json')
    def load(pid):
        try:
            if str(uuid.UUID(pid))!=pid: raise ValueError()
            p=projects/pid
            return p,json.loads((p/'project.json').read_text())
        except (ValueError,FileNotFoundError): raise HTTPException(404,'Spectra not found')
    def token():
        body=str(int(time.time())+7*86400)
        return body+'.'+hmac.new(secret.encode(),body.encode(),hashlib.sha256).hexdigest()
    def valid(value):
        try:
            body,sig=value.split('.')
            return int(body)>time.time() and hmac.compare_digest(sig,hmac.new(secret.encode(),body.encode(),hashlib.sha256).hexdigest())
        except (ValueError,AttributeError): return False
    def attach_wave(p, record, bump=True):
        if not any(c['id']=='wave' for c in record.get('manifest',{}).get('clips',[])):return False
        candidate=demo/'wave.mp4'
        if not record.get('existing_wave') or not candidate.exists():return False
        fingerprint=hashlib.sha256(candidate.read_bytes()).hexdigest()
        if record.get('wave_revision')==fingerprint:return False
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(candidate)],text=True,timeout=15))
        shutil.copyfile(candidate,p/'pack/clips/wave.mp4')
        clip=next(c for c in record['manifest']['clips'] if c['id']=='wave')
        clip.update(duration_seconds=duration)
        apply_timing(clip)
        record['wave_revision']=fingerprint
        if bump:record['manifest']['version']+=1
        return True

    def apply_timing(clip):
        timing=demo/'clip-timing.json'
        if timing.exists():
            details=json.loads(timing.read_text()).get(clip['id'],{})
            clip.update({k:v for k,v in details.items() if k in {'motion_onset_seconds','preparation'}})

    def attach_thumbs(p, record):
        if not any(c['id']=='thumbs_up' for c in record.get('manifest',{}).get('clips',[])):return False
        candidate=demo/'thumbs_up.mp4'
        if not record.get('existing_wave') or not candidate.exists():return False
        fingerprint=hashlib.sha256(candidate.read_bytes()).hexdigest()
        if record.get('thumbs_revision')==fingerprint:return False
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(candidate)],text=True,timeout=15))
        shutil.copyfile(candidate,p/'pack/clips/thumbs_up.mp4')
        clip=next(c for c in record['manifest']['clips'] if c['id']=='thumbs_up')
        clip.update(kind='existing',duration_seconds=duration,preparation='forward-reverse with original-photo endpoints')
        apply_timing(clip)
        record['thumbs_revision']=fingerprint
        record['manifest']['version']+=1
        return True

    def attach_idle(p, record):
        if 'motus_plan' in record and not any(m['id']=='idle_blink' for m in record['motus_plan']):return False
        candidate=demo/'idle_blink.mp4'
        if not record.get('existing_wave') or not candidate.exists():return False
        fingerprint=hashlib.sha256(candidate.read_bytes()).hexdigest()
        if record.get('idle_revision')==fingerprint:return False
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(candidate)],text=True,timeout=15))
        shutil.copyfile(candidate,p/'pack/clips/idle_blink.mp4')
        manifest=record['manifest']
        manifest['clips']=[c for c in manifest['clips'] if c['id']!='idle_blink']
        recipe=next((m for m in record.get('motus_plan',[]) if m['id']=='idle_blink'),{'name':'Blink','trigger':'idle'})
        manifest['clips'].append({'id':'idle_blink','label':recipe['name'],'file':'clips/idle_blink.mp4','kind':'existing','role':'idle' if recipe['trigger']=='idle' else 'reaction','duration_seconds':duration,'starts_at':'rest','ends_at':'rest','preparation':'Single forward blink with eased original-photo endpoints'})
        if recipe['trigger']=='idle':manifest['policy']['idle']={'clip_id':'idle_blink','delay_ms':4000}
        manifest['version']+=1
        record['idle_revision']=fingerprint
        return True

    def build(pid):
        p,record=load(pid)
        try:
            out=p/'pack'; (out/'clips').mkdir(parents=True,exist_ok=True)
            with Image.open(p/'source.png') as im:
                rest=ImageOps.pad(im.convert('RGB'),(832,1104),method=Image.Resampling.LANCZOS,color='#101916')
                rest.save(out/'rest.png')
            matching=False
            if (demo/'photo.jpg').exists() and (demo/'wave.mp4').exists():
                matching=photo_signature(p/'source.png')==photo_signature(demo/'photo.jpg')
            if matching:
                # Match the exact rest canvas used to repair the existing clip.
                with Image.open(demo/'photo.jpg') as im:
                    ImageOps.fit(ImageOps.exif_transpose(im).convert('RGB'),(832,1104),method=Image.Resampling.LANCZOS).save(out/'rest.png')
                shutil.copyfile(demo/'wave.mp4',out/'clips/wave.mp4')
            placeholder=out/'clips/thumbs_up.mp4'
            subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-i',str(out/'rest.png'),'-t','2','-r','24','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(placeholder)],check=True,capture_output=True,timeout=90)
            shutil.copyfile(placeholder,out/'still.mp4')
            if not matching: shutil.copyfile(placeholder,out/'clips/wave.mp4')
            manifest={'schema_version':1,'pack_id':pid,'name':record['name'],'version':1,'minimum_player_version':'0.1','canvas':{'width':832,'height':1104,'fps':24,'fit':'source-preserving pad' if not matching else 'existing wave canvas'},'rest':'rest.png','credits_spent':0,'mode':'placeholder','clips':[{'id':'wave','label':'Wave','file':'clips/wave.mp4','kind':'existing' if matching else 'placeholder','duration_seconds':151/24 if matching else 2,'starts_at':'rest','ends_at':'rest'},{'id':'thumbs_up','label':'Thumbs-up','file':'clips/thumbs_up.mp4','kind':'placeholder','duration_seconds':2,'starts_at':'rest','ends_at':'rest'}],'events':{'wave':'wave','thumbs_up':'thumbs_up'},'policy':{'interrupt':False,'cooldown_ms':600}}
            record.update(existing_wave=matching,manifest=manifest)
            attach_wave(p,record,bump=False)
            attach_thumbs(p,record)
            attach_idle(p,record)
            if 'motus_plan' in record:
                chosen=[];events={};idle=None
                for recipe in record['motus_plan']:
                    clip=next((dict(c) for c in manifest['clips'] if c['id']==recipe['id']),None)
                    if clip is None:
                        clip={'id':recipe['id'],'file':'clips/'+recipe['id']+'.mp4','kind':'placeholder','duration_seconds':2,'starts_at':'rest','ends_at':'rest'}
                        shutil.copyfile(out/'still.mp4',out/clip['file'])
                    clip['label']=recipe['name'];clip['role']='idle' if recipe['trigger']=='idle' else 'reaction'
                    chosen.append(clip)
                    if recipe['trigger']=='idle':idle=recipe['id']
                    elif recipe['trigger']!='manual':events[recipe['trigger']]=recipe['id']
                manifest['clips']=chosen;manifest['events']=events
                manifest['policy'].pop('idle',None)
                if idle:manifest['policy']['idle']={'clip_id':idle,'delay_ms':4000}
                manifest['motus']=[{k:r[k] for k in ('id','name','revision','token_price')} for r in record['motus_plan']]
            (out/'still.mp4').unlink(missing_ok=True)
            make_pack(out,manifest)
            with lock:
                record.update(status='needs_review'); save(p,record)
        except Exception:
            with lock:
                _,record=load(pid); record.update(status='failed',error='Preparation stopped. Retry this local placeholder job; no credits were spent.');save(p,record)
    @asynccontextmanager
    async def lifespan(app):
        for p in projects.glob('*/project.json'):
            record=json.loads(p.read_text())
            if record['status']=='preparing':
                record.update(status='failed',error='Server restarted during preparation. Retry to resume; no credits were spent.');save(p.parent,record)
            elif record['status'] in {'needs_review','ready'} and any([attach_wave(p.parent,record),attach_thumbs(p.parent,record),attach_idle(p.parent,record)]):
                # A changed reaction must be reviewed again before a new export.
                make_pack(p.parent/'pack',record['manifest'])
                record['status']='needs_review';save(p.parent,record)
        yield
        pool.shutdown(wait=True)
    app=FastAPI(lifespan=lifespan)
    app.state.root=root
    mount_motus(app,motus)
    @app.middleware('http')
    async def auth(request,call_next):
        path=request.url.path
        if request.method in {'POST','PUT','DELETE','PATCH'}:
            origin=request.headers.get('origin')
            if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
                return JSONResponse({'detail':'Cross-origin request refused'},403)
        if path not in {'/login','/health'} and not valid(request.cookies.get('studio')):
            if path.startswith('/api/') or path.startswith('/media/'):
                return JSONResponse({'detail':'Please sign in again.'},401)
            return RedirectResponse('/login',status_code=303)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['Cache-Control']='no-store'
        response.headers['X-Frame-Options']='SAMEORIGIN'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; frame-src 'self'; connect-src 'self'; frame-ancestors 'self'; base-uri 'none'; form-action 'self'"
        # Only the bundled recognizer worker compiles WebAssembly. Keep the
        # document policy and JavaScript eval restrictions unchanged.
        if path.endswith('/runtime/gesture-worker.js') and response.status_code==200:
            response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; object-src 'none'"
        return response
    @app.get('/health')
    def health(): return {'status':'ok','mode':'placeholder','generation_enabled':False}
    @app.get('/login',response_class=HTMLResponse)
    def login(): return (STATIC/'login.html').read_text()
    @app.post('/login')
    async def sign_in(request:Request,password:str=Form(...)):
        key=request.client.host if request.client else 'unknown'
        now=time.time(); recent=[t for t in attempts.get(key,[]) if now-t<60]
        if len(recent)>=10: raise HTTPException(429,'Please wait a minute before trying again.')
        if not secrets.compare_digest(password,secret):
            attempts[key]=recent+[now]
            return HTMLResponse((STATIC/'login.html').read_text().replace('Enter your workshop password.','Incorrect password. Please try again.'),status_code=401)
        attempts.pop(key,None)
        response=RedirectResponse('/',303);response.set_cookie('studio',token(),httponly=True,secure=secure,samesite='strict',max_age=7*86400);return response
    @app.post('/logout')
    def logout():
        response=RedirectResponse('/login',303);response.delete_cookie('studio');return response
    @app.get('/',response_class=HTMLResponse)
    def index():return (STATIC/'index.html').read_text()
    @app.get('/api/projects')
    def listing():
        result=[json.loads(p.read_text()) for p in projects.glob('*/project.json')]
        return sorted(result,key=lambda r:r['created_at'],reverse=True)
    @app.get('/api/demo')
    def demo_available():return {'available':(demo/'photo.jpg').exists() and (demo/'wave.mp4').exists()}
    async def ingest(raw,name,selection=None):
        plan=motus.select(selection)
        if len(raw)>MAX_UPLOAD:raise HTTPException(413,'Photo must be under 15 MB.')
        try:
            with Image.open(io.BytesIO(raw)) as im:
                if im.width*im.height>40_000_000:raise ValueError('Photo must be under 40 megapixels.')
                photo=ImageOps.exif_transpose(im).convert('RGB');photo.load()
        except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError):
            raise HTTPException(400,'Upload a valid JPG, PNG or WebP photo under 40 megapixels.')
        with lock:
            pending=sum(json.loads(p.read_text())['status']=='preparing' for p in projects.glob('*/project.json'))
            if pending>=3:raise HTTPException(429,'Three spectras are already preparing. Please wait.')
            if len(list(projects.iterdir()))>=100:raise HTTPException(409,'Workshop limit reached (100 spectras). Export and archive data before adding more.')
            pid=str(uuid.uuid4());p=projects/pid;p.mkdir()
            photo.save(p/'source.png')
            record={'id':pid,'name':(name.strip() or 'Untitled spectra')[:80],'status':'preparing','created_at':time.time(),'credits_spent':0,'motus_plan':plan,'planned_tokens':sum(m['token_price'] for m in plan),'tokens_charged':0,'generation_enabled':False}
            save(p,record)
        pool.submit(build,pid)
        return record
    @app.post('/api/projects',status_code=202)
    async def upload(photo:UploadFile=File(...),name:str=Form('My spectra'),motus_ids:str|None=Form(None)):
        raw=await photo.read(MAX_UPLOAD+1);await photo.close();return await ingest(raw,name,motus_ids)
    @app.post('/api/demo',status_code=202)
    async def import_demo():
        if not (demo/'photo.jpg').exists():raise HTTPException(404,'Example is not installed.')
        return await ingest((demo/'photo.jpg').read_bytes(),'Karol at Plitvice')
    @app.get('/api/projects/{pid}')
    def project(pid:str):return load(pid)[1]
    @app.post('/api/projects/{pid}/retry',status_code=202)
    def retry(pid:str):
        with lock:
            p,r=load(pid)
            if r['status']!='failed':raise HTTPException(409,'Only failed preparation can be retried.')
            r.update(status='preparing');r.pop('error',None);save(p,r)
        pool.submit(build,pid);return r
    @app.post('/api/projects/{pid}/approve')
    def approve(pid:str):
        with lock:
            p,r=load(pid)
            if r['status'] not in {'needs_review','ready'}:raise HTTPException(409,'Preview must finish preparing first.')
            r.update(status='ready');save(p,r)
        return r
    @app.get('/api/projects/{pid}/download')
    def download(pid:str):
        p,r=load(pid)
        if r['status']!='ready':raise HTTPException(409,'Approve the preview before downloading.')
        with lock:
            expected=(STATIC/'preview.html').read_text().replace('__MANIFEST__',json.dumps(r['manifest']).replace('<','\\u003c'))
            if (p/'pack/preview.html').read_text()!=expected:
                make_pack(p/'pack',r['manifest'])
        return FileResponse(p/'portrait.zip',media_type='application/zip',filename='spectramotus-'+pid[:8]+'.zip')
    @app.get('/media/{pid}/{path:path}')
    def media(pid:str,path:str):
        p,r=load(pid)
        if r['status'] not in {'needs_review','ready'}:raise HTTPException(409,'Still preparing')
        allowed={'rest.png','preview.html','manifest.json'}|{c['file'] for c in r['manifest']['clips']}
        if path in runtime_files():return FileResponse(STATIC/path)
        if path not in allowed:raise HTTPException(404)
        if path=='preview.html':
            return HTMLResponse((STATIC/'preview.html').read_text().replace('__MANIFEST__',json.dumps(r['manifest']).replace('<','\\u003c')))
        return FileResponse(p/'pack'/path)
    return app
