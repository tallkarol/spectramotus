"""Owner-managed movement recipes. Configuration only; never calls a generator."""
from pathlib import Path
import hashlib
import json
import subprocess
import threading
import time
import uuid
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from .navigation import render_page
from .database import motus, revisions
from sqlalchemy import select, insert, update

MAX_VIDEO=64*1024*1024
SEEDS=[
    ('wave','Wave','A friendly wave, then back to the photograph.','wave'),
    ('thumbs_up','Thumbs-up','A little encouragement, then back to rest.','thumbs_up'),
    ('idle_blink','A quiet moment','A subtle blink while the spectrum waits.','idle'),
]

class MotusLibrary:
    def __init__(self, root, store):
        self.store=store
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.assets=self.root/'assets';self.assets.mkdir(exist_ok=True)
        self.lock=threading.RLock()
        for ident,name,description,trigger in SEEDS:
            with store.engine.connect() as c:exists=c.execute(select(motus.c.id).where(motus.c.id==ident)).first()
            if not exists:
                self.write(dict(id=ident,name=name,description=description,trigger=trigger,prompt='',instructions='',enabled=True,token_price=0,revision=1,driving_video=None,updated_at=time.time()))

    def write(self, record):
        with self.store.engine.begin() as c:
            previous=c.execute(select(motus).where(motus.c.id==record['id']).with_for_update()).mappings().first()
            if previous and record['revision']!=previous['data']['revision']+1:raise HTTPException(409,'This movement changed in another tab. Reload before saving.')
            record.setdefault('publication','published' if record['enabled'] else 'disabled')
            published=record['revision'] if record['publication']=='published' else previous['published_revision'] if previous else None
            values=dict(data=record,published_revision=published)
            if previous:c.execute(update(motus).where(motus.c.id==record['id']).values(**values))
            else:c.execute(insert(motus).values(id=record['id'],**values))
            c.execute(insert(revisions).values(motus_id=record['id'],revision=record['revision'],data=record))

    def get(self, ident):
        with self.store.engine.connect() as c:
            record=c.execute(select(motus.c.data).where(motus.c.id==ident)).scalar_one_or_none()
            if record is None:raise HTTPException(404,'Movement not found.')
            return record

    def listing(self):
        with self.store.engine.connect() as c:
            return sorted(c.execute(select(motus.c.data)).scalars(),key=lambda r:(r['id'] not in {s[0] for s in SEEDS},r['name'].lower()))

    def published(self):
        with self.store.engine.connect() as c:
            rows=c.execute(select(motus.c.data,revisions.c.data.label('published')).join(revisions,(revisions.c.motus_id==motus.c.id)&(revisions.c.revision==motus.c.published_revision))).mappings()
            return [r['published'] for r in rows if r['data'].get('publication')!='disabled']

    def history(self,ident):
        self.get(ident)
        with self.store.engine.connect() as c:return list(c.execute(select(revisions.c.data).where(revisions.c.motus_id==ident).order_by(revisions.c.revision.desc())).scalars())

    def select(self, value=None):
        available={r['id']:r for r in self.published()}
        if value is None:ids=[s[0] for s in SEEDS if s[0] in available]
        else:
            try:ids=json.loads(value)
            except (ValueError,TypeError):raise HTTPException(400,'Choose valid movements.')
        if not isinstance(ids,list) or not 1<=len(ids)<=12 or any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids):
            raise HTTPException(400,'Choose between 1 and 12 different movements.')
        with self.lock:
            if any(i not in available for i in ids):raise HTTPException(409,'A selected movement is no longer published. Refresh and choose again.')
            rows=[available[i] for i in ids]
        if any(not r['enabled'] for r in rows):raise HTTPException(409,'A selected movement is no longer available. Refresh and choose again.')
        triggers=[r['trigger'] for r in rows if r['trigger']!='manual']
        if len(triggers)!=len(set(triggers)):raise HTTPException(400,'Choose one movement for each gesture or idle slot.')
        # Immutable project snapshot: later recipe changes never silently change this plan.
        return rows


def mount_motus(app, library):
    router=APIRouter()

    @router.get('/admin',response_class=HTMLResponse)
    def admin(request:Request):return render_page(request,'admin.html')

    @router.get('/api/motus')
    def catalog():
        return {'billing_enabled':False,'generation_enabled':False,'items':[
            {k:r[k] for k in ('id','name','description','trigger','token_price','revision')}
            for r in library.published()]}

    @router.get('/api/admin/motus')
    def listing():return {'items':library.listing(),'published_revisions':{r['id']:r['revision'] for r in library.published()},'billing_enabled':False,'generation_enabled':False}

    @router.get('/api/admin/motus/{ident}/revisions')
    def history(ident:str):return {'items':library.history(ident)}

    @router.get('/api/admin/motus/{ident}/video')
    def video(ident:str):
        row=library.get(ident);name=row['driving_video']
        if not name:raise HTTPException(404,'No driving video attached.')
        return FileResponse(library.assets/name,media_type='video/mp4')

    @router.post('/api/admin/motus')
    async def save_recipe(name:str=Form(...),description:str=Form(''),prompt:str=Form(''),instructions:str=Form(''),trigger:str=Form('manual'),enabled:bool=Form(False),token_price:int=Form(0),ident:str=Form(''),revision:int=Form(0),remove_video:bool=Form(False),publication:str=Form(''),markdown:UploadFile|None=File(None),driving_video:UploadFile|None=File(None)):
        name=name.strip();description=description.strip()
        publication=publication or ('published' if enabled else 'disabled')
        if publication not in {'draft','published','disabled'}:raise HTTPException(400,'Choose draft, published, or disabled.')
        enabled=publication!='disabled'
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
            record=dict(id=ident,name=name,publication=publication,description=description,prompt=prompt,instructions=instructions,trigger=trigger,enabled=enabled,token_price=token_price,revision=previous['revision']+1,driving_video=asset or (None if remove_video else previous['driving_video']),updated_at=time.time())
            library.write(record)
        return record
    app.include_router(router)
