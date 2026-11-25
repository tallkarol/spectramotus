"""Invite-only email sign-in, revocable sessions, and owner/admin/beta access."""
import os
import re
import secrets
import time
import uuid
from urllib.parse import urlparse
import httpx
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, insert, update, delete
from .database import users, emails, links, sessions, limits, audit, fingerprint
from .packs import STATIC
from .email_messages import email_message

LINK_TTL=15*60
SESSION_TTL=7*24*60*60
INVITE_TTL=7*24*60*60
GENERIC='If this email has access, a sign-in link is on its way. Check your inbox and spam folder.'

def email_address(raw):
    address=raw.strip().lower()
    if len(address)>254 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+',address):raise HTTPException(400,'Enter a valid email address.')
    return address

def has_access(user,now):
    return user and (user['status']=='active' or (user['status']=='invited' and (user['invite_expires_at'] or 0)>now))

class ResendMailer:
    def __init__(self):
        self.key=os.getenv('RESEND_API_KEY');self.sender=os.getenv('RESEND_FROM_EMAIL')
        if not self.key or not self.sender:raise RuntimeError('Configure Resend before enabling email sign-in.')
    def __call__(self,address,url,kind='signin'):
        response=httpx.post('https://api.resend.com/emails',headers={'Authorization':'Bearer '+self.key,'Idempotency-Key':fingerprint(kind+url)},json={'from':'Spectramotus <'+self.sender+'>','to':[address],**email_message(url,kind)},timeout=15)
        if response.status_code>=400:raise RuntimeError('Sign-in email delivery failed.')

class Accounts:
    def __init__(self,store,app_url,secure,mailer):
        self.store=store;self.app_url=app_url.rstrip('/');self.secure=secure;self.mailer=mailer
        parsed=urlparse(self.app_url)
        if parsed.scheme not in {'http','https'} or not parsed.netloc or parsed.path not in {'','/'} or parsed.query or parsed.fragment:raise RuntimeError('STUDIO_APP_URL must be an origin without a path.')
        if secure and parsed.scheme!='https':raise RuntimeError('Production email sign-in requires HTTPS.')
        self.cookie='__Host-spectramotus' if secure else 'spectramotus'

    def session(self,raw):
        if not raw or len(raw)>100:return None
        with self.store.engine.connect() as c:
            row=c.execute(select(users).join(sessions,sessions.c.user_id==users.c.id).where(sessions.c.hash==fingerprint(raw),sessions.c.expires_at>int(time.time()),users.c.status=='active')).mappings().first()
            return dict(row) if row else None

    def throttle(self,key,count,period):
        now=int(time.time());values={'key':fingerprint(key+':'+str(now//period)),'window':now,'count':1}
        if self.store.engine.dialect.name=='postgresql':
            from sqlalchemy.dialects.postgresql import insert as upsert
        else:
            from sqlalchemy.dialects.sqlite import insert as upsert
        with self.store.engine.begin() as c:
            c.execute(delete(limits).where(limits.c.window<now-86400))
            n=c.execute(upsert(limits).values(**values).on_conflict_do_update(index_elements=['key'],set_={'count':limits.c.count+1}).returning(limits.c.count)).scalar_one()
        return n<=count

    def send(self,address):
        user=self.store.by_email(address);now=int(time.time())
        if not has_access(user,now):return False
        raw=secrets.token_urlsafe(32);hashed=fingerprint(raw)
        with self.store.engine.begin() as c:
            c.execute(delete(links).where(links.c.expires_at<now-86400))
            c.execute(delete(sessions).where(sessions.c.expires_at<now))
            c.execute(insert(links).values(hash=hashed,user_id=user['id'],email=address,expires_at=now+LINK_TTL))
        try:self.mailer(address,self.app_url+'/auth/verify#token='+raw,kind='invite' if user['status']=='invited' else 'signin')
        except Exception:
            with self.store.engine.begin() as c:
                c.execute(delete(links).where(links.c.hash==hashed));self.store.event(c,None,'sign_in_delivery_failed',user['id'])
            return False
        return True

    def inspect(self,raw):
        if not raw or len(raw)>100:raise HTTPException(400,'This sign-in link is invalid or expired. Request a new one.')
        now=int(time.time())
        with self.store.engine.connect() as c:
            row=c.execute(select(links.c.email,users.c.status,users.c.invite_expires_at).join(users,users.c.id==links.c.user_id).where(links.c.hash==fingerprint(raw),links.c.used_at.is_(None),links.c.expires_at>now)).mappings().first()
            if not has_access(row,now):raise HTTPException(400,'This sign-in link is invalid or expired. Request a new one.')
            return row['email']

    def consume(self,raw):
        if not raw or len(raw)>100:raise HTTPException(400,'This sign-in link is invalid or expired. Request a new one.')
        now=int(time.time());hashed=fingerprint(raw)
        with self.store.engine.begin() as c:
            link=c.execute(select(links).where(links.c.hash==hashed)).mappings().first()
            if not link:raise HTTPException(400,'This sign-in link is invalid or expired. Request a new one.')
            user=c.execute(select(users).where(users.c.id==link['user_id']).with_for_update()).mappings().first()
            if not has_access(user,now):raise HTTPException(400,'This sign-in link is invalid or expired. Request a new one.')
            # Conditional atomic update: concurrent clicks can create only one session.
            consumed=c.execute(update(links).where(links.c.hash==hashed,links.c.used_at.is_(None),links.c.expires_at>now).values(used_at=now).returning(links.c.user_id)).scalar_one_or_none()
            if not consumed:raise HTTPException(400,'This sign-in link is invalid or expired. Request a new one.')
            raw_session=secrets.token_urlsafe(32)
            c.execute(insert(sessions).values(hash=fingerprint(raw_session),user_id=user['id'],expires_at=now+SESSION_TTL))
            c.execute(update(users).where(users.c.id==user['id']).values(status='active'))
            self.store.event(c,user['id'],'signed_in',user['id'])
        return raw_session

    def grant(self,address,role,actor):
        if role not in {'admin','beta'} or (role=='admin' and actor['role']!='owner'):raise HTTPException(403,'Only the owner can grant admin access.')
        with self.store.engine.begin() as c:
            existing=self.store.by_email(address,c)
            if existing:raise HTTPException(409,'That email already has an account. Manage its existing access below.')
            ident=str(uuid.uuid4());now=int(time.time())
            c.execute(insert(users).values(id=ident,email=address,role=role,status='invited',invite_expires_at=now+INVITE_TTL,created_at=now))
            c.execute(insert(emails).values(email=address,user_id=ident));self.store.event(c,actor['id'],'access_prepared',ident)
        return ident


def mount_accounts(app,accounts):
    router=APIRouter()
    def require_admin(request):
        if request.state.user['role'] not in {'owner','admin'}:raise HTTPException(403,'Admin access required.')
        return request.state.user

    @router.get('/api/me')
    def me(request:Request):return {'id':request.state.user['id'],'email':request.state.user['email'],'role':request.state.user['role']}

    @router.post('/auth/request')
    def request_link(request:Request,email:str=Form(...)):
        address=email_address(email)
        ip=request.client.host if request.client else 'unknown'
        if accounts.throttle('request-ip:'+ip,15,3600) and accounts.throttle('request-email:'+address,3,900):accounts.send(address)
        return {'message':GENERIC}

    @router.get('/auth/verify',response_class=HTMLResponse)
    def verify():return (STATIC/'verify.html').read_text()

    @router.post('/auth/inspect')
    def inspect(request:Request,token:str=Form(...)):
        if not accounts.throttle('inspect:'+str(request.client.host),60,900):raise HTTPException(429,'Please wait before trying again.')
        return {'email':accounts.inspect(token)}

    @router.post('/auth/consume')
    def consume(request:Request,token:str=Form(...)):
        if not accounts.throttle('consume:'+str(request.client.host),30,900):raise HTTPException(429,'Please wait before trying again.')
        raw=accounts.consume(token)
        from fastapi.responses import JSONResponse
        response=JSONResponse({'redirect':'/studio'})
        response.set_cookie(accounts.cookie,raw,max_age=SESSION_TTL,secure=accounts.secure,httponly=True,samesite='lax',path='/')
        response.delete_cookie('studio',path='/')
        return response

    @router.get('/admin/users',response_class=HTMLResponse)
    def team(request:Request):require_admin(request);return (STATIC/'users.html').read_text()

    @router.get('/api/admin/users')
    def listing(request:Request):
        actor=require_admin(request)
        with accounts.store.engine.connect() as c:
            rows=[dict(r) for r in c.execute(select(users).order_by(users.c.created_at)).mappings()]
            for r in rows:r['aliases']=[v for v in c.execute(select(emails.c.email).where(emails.c.user_id==r['id'])).scalars()]
        return {'items':rows,'can_grant_admin':actor['role']=='owner'}

    @router.post('/api/admin/users')
    def grant(request:Request,email:str=Form(...),role:str=Form('beta')):
        actor=require_admin(request);ident=accounts.grant(email_address(email),role,actor)
        return {'id':ident,'message':'Access prepared. No email has been sent.'}

    @router.post('/api/admin/users/{ident}/status')
    def change_status(ident:str,request:Request,status:str=Form(...)):
        actor=require_admin(request)
        if status not in {'invited','disabled'}:raise HTTPException(400,'Choose invited or disabled.')
        with accounts.store.engine.begin() as c:
            target=c.execute(select(users).where(users.c.id==ident).with_for_update()).mappings().first()
            if not target:raise HTTPException(404,'Account not found.')
            if target['role']=='owner' or target['id']==actor['id'] or (target['role']=='admin' and actor['role']!='owner'):raise HTTPException(403,'You cannot change this account.')
            c.execute(update(users).where(users.c.id==ident).values(status=status,invite_expires_at=int(time.time())+INVITE_TTL if status=='invited' else None))
            c.execute(delete(sessions).where(sessions.c.user_id==ident));c.execute(delete(links).where(links.c.user_id==ident))
            accounts.store.event(c,actor['id'],'access_'+status,ident)
        return {'ok':True}

    @router.post('/api/admin/users/{ident}/send')
    def send_invite(ident:str,request:Request):
        actor=require_admin(request)
        with accounts.store.engine.begin() as c:
            target=c.execute(select(users).where(users.c.id==ident).with_for_update()).mappings().first()
            if not target:raise HTTPException(404,'Account not found.')
            if target['status']=='disabled' or target['role']=='owner' or (target['role']=='admin' and actor['role']!='owner'):raise HTTPException(403,'You cannot send this invitation.')
            if target['status']=='invited':c.execute(update(users).where(users.c.id==ident).values(invite_expires_at=int(time.time())+INVITE_TTL))
        if not accounts.throttle('invite:'+ident,3,900):raise HTTPException(429,'Please wait before sending another link.')
        if not accounts.send(target['email']):raise HTTPException(503,'Email could not be sent. Access is saved; please retry later.')
        return {'message':'Sign-in invitation sent. The link expires in 15 minutes.'}
    app.include_router(router)
