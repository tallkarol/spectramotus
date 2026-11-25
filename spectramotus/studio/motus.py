"""Owner-managed movement recipes. Configuration only; never calls a generator."""
from pathlib import Path
import hashlib
import json
import subprocess
import threading
import time
import uuid
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from .packs import STATIC

MAX_VIDEO=64*1024*1024
SEEDS=[
    ('wave','Wave','A friendly wave, then back to the photograph.','wave'),
    ('thumbs_up','Thumbs-up','A little encouragement, then back to rest.','thumbs_up'),
    ('idle_blink','A quiet moment','A subtle blink while the spectra waits.','idle'),
]

class MotusLibrary:
    def __init__(self, root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.assets=self.root/'assets';self.assets.mkdir(exist_ok=True)
        self.lock=threading.RLock()
        for ident,name,description,trigger in SEEDS:
            path=self.root/(ident+'.json')
            if not path.exists():
                self.write(dict(id=ident,name=name,description=description,trigger=trigger,prompt='',instructions='',enabled=True,token_price=0,revision=1,driving_video=None,updated_at=time.time()))

    def write(self, record):
        target=self.root/(record['id']+'.json');tmp=target.with_suffix('.tmp')
        tmp.write_text(json.dumps(record,indent=2));tmp.replace(target)

    def get(self, ident):
        if not isinstance(ident,str) or not ident.replace('_','').replace('-','').isalnum() or len(ident)>80:
            raise HTTPException(404,'Movement not found.')
        try:return json.loads((self.root/(ident+'.json')).read_text())
        except FileNotFoundError:raise HTTPException(404,'Movement not found.')

    def listing(self):
        with self.lock:
            return sorted([json.loads(p.read_text()) for p in self.root.glob('*.json')],key=lambda r:(r['id'] not in {s[0] for s in SEEDS},r['name'].lower()))

    def select(self, value=None):
        if value is None:ids=[s[0] for s in SEEDS if self.get(s[0])['enabled']]
        else:
            try:ids=json.loads(value)
            except (ValueError,TypeError):raise HTTPException(400,'Choose valid movements.')
        if not isinstance(ids,list) or not 1<=len(ids)<=12 or any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids):
            raise HTTPException(400,'Choose between 1 and 12 different movements.')
        with self.lock:
            rows=[self.get(i) for i in ids]
        if any(not r['enabled'] for r in rows):raise HTTPException(409,'A selected movement is no longer available. Refresh and choose again.')
        triggers=[r['trigger'] for r in rows if r['trigger']!='manual']
        if len(triggers)!=len(set(triggers)):raise HTTPException(400,'Choose one movement for each gesture or idle slot.')
        # Immutable project snapshot: later recipe changes never silently change this plan.
        return rows


def mount_motus(app, library):
    router=APIRouter()

    @router.get('/admin',response_class=HTMLResponse)
    def admin():return (STATIC/'admin.html').read_text()

    @router.get('/api/motus')
    def catalog():
        return {'billing_enabled':False,'generation_enabled':False,'items':[
            {k:r[k] for k in ('id','name','description','trigger','token_price','revision')}
            for r in library.listing() if r['enabled']]}

    @router.get('/api/admin/motus')
    def listing():return {'items':library.listing(),'billing_enabled':False,'generation_enabled':False}

    @router.get('/api/admin/motus/{ident}/video')
    def video(ident:str):
        row=library.get(ident);name=row['driving_video']
        if not name:raise HTTPException(404,'No driving video attached.')
        return FileResponse(library.assets/name,media_type='video/mp4')

    @router.post('/api/admin/motus')
    async def save_recipe(name:str=Form(...),description:str=Form(''),prompt:str=Form(''),instructions:str=Form(''),trigger:str=Form('manual'),enabled:bool=Form(False),token_price:int=Form(0),ident:str=Form(''),revision:int=Form(0),remove_video:bool=Form(False),markdown:UploadFile|None=File(None),driving_video:UploadFile|None=File(None)):
        name=name.strip();description=description.strip()
        if not name or len(name)>80 or len(description)>300 or len(prompt)>12000 or len(instructions)>64000:
            raise HTTPException(400,'Use a name under 80 characters, description under 300, prompt under 12,000 and instructions under 64,000.')
        if trigger not in {'wave','thumbs_up','idle','manual'} or not 0<=token_price<=100000:
            raise HTTPException(400,'Choose a valid trigger and token price (0–100,000).')
        if markdown and markdown.filename:
            raw=await markdown.read(64001);await markdown.close()
            if len(raw)>64000 or Path(markdown.filename).suffix.lower() not in {'.md','.markdown'}:raise HTTPException(400,'Use a Markdown file under 64 KB.')
            try:instructions=raw.decode('utf-8')
            except UnicodeDecodeError:raise HTTPException(400,'Markdown must use UTF-8 text.')
        asset=None
        if driving_video and driving_video.filename:
            if Path(driving_video.filename).suffix.lower()!='.mp4':raise HTTPException(400,'Use an H.264 MP4 driving video, up to 60 seconds and 64 MB.')
            raw=await driving_video.read(MAX_VIDEO+1);await driving_video.close()
            if len(raw)>MAX_VIDEO:raise HTTPException(413,'Driving video must be under 64 MB.')
            temp=library.assets/(str(uuid.uuid4())+'.tmp')
            try:
                temp.write_bytes(raw)
                info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(temp)],stderr=subprocess.DEVNULL,timeout=15))
                streams=[s for s in info['streams'] if s['codec_type']=='video']
                if not streams or streams[0]['codec_name']!='h264' or not 0<float(info['format']['duration'])<=60 or streams[0]['width']*streams[0]['height']>4096*2160:raise ValueError()
                asset=hashlib.sha256(raw).hexdigest()+'.mp4'
                temp.replace(library.assets/asset)
            except (ValueError,KeyError,subprocess.SubprocessError):raise HTTPException(400,'Use a valid H.264 MP4 driving video, up to 60 seconds and 4K.')
            finally:temp.unlink(missing_ok=True)
        with library.lock:
            if ident:
                previous=library.get(ident)
                if previous['revision']!=revision:raise HTTPException(409,'This movement changed in another tab. Reload it before saving.')
            else:
                if len(library.listing())>=100:raise HTTPException(409,'Library limit reached (100 movements).')
                ident=str(uuid.uuid4());previous={'revision':0,'driving_video':None}
            record=dict(id=ident,name=name,description=description,prompt=prompt,instructions=instructions,trigger=trigger,enabled=enabled,token_price=token_price,revision=previous['revision']+1,driving_video=asset or (None if remove_video else previous['driving_video']),updated_at=time.time())
            library.write(record)
        return record
    app.include_router(router)
