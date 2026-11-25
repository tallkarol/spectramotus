import io
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse,parse_qs
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text,select,update
from sqlalchemy.engine import make_url
from spectramotus.studio.app import create_app
from spectramotus.studio.database import users,links,sessions,spectra,Store,fingerprint

@pytest.fixture
def studio(tmp_path):
    url=os.getenv('TEST_DATABASE_URL')
    schema=None;admin=None
    if url:
        admin=create_engine(url,hide_parameters=True);schema='test_'+uuid.uuid4().hex
        with admin.begin() as c:c.execute(text('CREATE SCHEMA '+schema))
        url=make_url(url).update_query_dict({'options':'-csearch_path='+schema}).render_as_string(hide_password=False)
    else:url='sqlite:///'+str(tmp_path/'accounts.sqlite')
    messages=[]
    app=create_app(data_dir=tmp_path/'data',database_url=url,auth_mode='email',owner_emails=['owner@example.com','alias@example.com'],mailer=lambda email,url,kind='signin':messages.append((email,url,kind)))
    with TestClient(app) as client:yield app,client,messages
    if admin:
        with admin.begin() as c:c.execute(text('DROP SCHEMA '+schema+' CASCADE'))
        admin.dispose()

def token(messages):return parse_qs(urlparse(messages[-1][1]).fragment)['token'][0]
def login(client,messages,email='owner@example.com'):
    assert client.post('/auth/request',data={'email':email}).status_code==200
    raw=token(messages)
    assert client.post('/auth/consume',data={'token':raw}).status_code==200
    return raw

def make_photo():
    f=io.BytesIO();Image.new('RGB',(120,160),'green').save(f,format='PNG');return f.getvalue()
def wait(client,pid):
    for _ in range(150):
        r=client.get('/api/projects/'+pid).json()
        if r['status']!='preparing':return r
        time.sleep(.1)
    raise AssertionError('Preparation timed out')

def test_email_alias_single_use_scanner_safe_and_old_password_disabled(studio):
    app,c,m=studio
    assert 'Email me a sign-in link' in c.get('/login').text
    assert c.post('/login',data={'password':'old'}).status_code==410
    unknown=c.post('/auth/request',data={'email':'unknown@example.com'})
    known=c.post('/auth/request',data={'email':'alias@example.com'})
    assert unknown.json()==known.json() and len(m)==1
    raw=token(m)
    assert c.get('/auth/verify').status_code==200
    assert c.post('/auth/inspect',data={'token':raw}).json()['email']=='alias@example.com'
    assert c.get('/api/me').status_code==401
    response=c.post('/auth/consume',data={'token':raw});assert response.status_code==200
    assert 'httponly' in response.headers['set-cookie'].lower()
    assert c.get('/api/me').json()['email']=='owner@example.com'
    assert c.post('/auth/consume',data={'token':raw}).status_code==400
    with app.state.store.engine.connect() as db:
        assert len(list(db.execute(select(users))))==1
        assert db.execute(select(links.c.hash)).scalar_one()!=raw
    c.post('/logout');assert c.get('/api/me').status_code==401

def test_invite_permissions_and_immediate_revocation(studio):
    app,c,m=studio;login(c,m)
    admin=c.post('/api/admin/users',data={'email':'admin@example.com','role':'admin'}).json()['id']
    beta=c.post('/api/admin/users',data={'email':'beta@example.com','role':'beta'}).json()['id']
    assert len(m)==1 # Preparing access never sends email.
    beta_client=TestClient(app);raw=login(beta_client,m,'beta@example.com')
    assert beta_client.get('/admin').status_code==403
    assert beta_client.get('/api/admin/users').status_code==403
    assert beta_client.get('/api/admin/motus').status_code==403
    admin_client=TestClient(app);login(admin_client,m,'admin@example.com')
    assert admin_client.get('/api/admin/motus').status_code==200
    assert admin_client.post('/api/admin/users',data={'email':'elevated@example.com','role':'admin'}).status_code==403
    assert admin_client.post('/api/admin/users/'+app.state.owner['id']+'/status',data={'status':'disabled'}).status_code==403
    assert c.post('/api/admin/users/'+beta+'/status',data={'status':'disabled'}).status_code==200
    assert beta_client.get('/api/projects').status_code==401
    c.post('/api/admin/users/'+admin+'/status',data={'status':'disabled'})
    assert admin_client.get('/api/projects').status_code==401

def test_beta_spectra_are_private_and_admin_assets_hidden(studio):
    app,c,m=studio;login(c,m)
    c.post('/api/admin/users',data={'email':'beta@example.com','role':'beta'})
    other=TestClient(app);login(other,m,'beta@example.com')
    pid=c.post('/api/projects',files={'photo':('p.png',make_photo())}).json()['id'];wait(c,pid)
    for path in ['/api/projects/'+pid,'/media/'+pid+'/rest.png','/media/'+pid+'/preview.html','/media/'+pid+'/runtime/camera.js','/api/projects/'+pid+'/download']:
        assert other.get(path).status_code==404,path
    assert other.post('/api/projects/'+pid+'/approve').status_code==404
    assert other.get('/api/projects').json()==[]
    assert other.get('/api/demo').json()['available'] is False
    assert other.post('/api/demo').status_code==403
    own=other.post('/api/projects',files={'photo':('p.png',make_photo())});assert own.status_code==202,own.text
    assert 'prompt' not in own.json()['motus_plan'][0]
    own_id=own.json()['id'];r=wait(other,own_id)
    assert r['status']=='needs_review' and r['owner_id']!=app.state.owner['id']
    assert 'instructions' not in r['motus_plan'][0]
    assert other.get('/media/'+own_id+'/rest.png').status_code==200
    assert len(c.get('/api/projects').json())==2

def test_expiry_rate_limits_and_cross_origin(studio):
    app,c,m=studio
    for _ in range(5):c.post('/auth/request',data={'email':'owner@example.com'})
    assert len(m)==3
    raw=token(m)
    with app.state.store.engine.begin() as db:db.execute(update(links).values(expires_at=int(time.time())-1))
    assert c.post('/auth/consume',data={'token':raw}).status_code==400
    assert c.post('/auth/request',data={'email':'owner@example.com'},headers={'origin':'https://evil.example'}).status_code==403
    assert c.get('/auth/verify').headers['referrer-policy']=='no-referrer'

def test_concurrent_link_consumption_creates_one_session(studio):
    app,c,m=studio;c.post('/auth/request',data={'email':'owner@example.com'});raw=token(m)
    def consume():
        try:app.state.accounts.consume(raw);return True
        except Exception:return False
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:consume(),range(2)))
    assert results.count(True)==1
    with app.state.store.engine.connect() as db:assert len(list(db.execute(select(sessions))))==1

def test_drafts_preserve_published_recipe_and_keep_history(studio):
    app,c,m=studio;login(c,m)
    original=next(r for r in c.get('/api/admin/motus').json()['items'] if r['id']=='wave')
    data={**original,'ident':'wave','name':'Draft wave','prompt':'Private draft','publication':'draft'}
    assert c.post('/api/admin/motus',data=data).status_code==200
    catalog=c.get('/api/motus').json()['items'];assert next(r for r in catalog if r['id']=='wave')['name']=='Wave'
    data.update(revision=2,publication='published')
    assert c.post('/api/admin/motus',data=data).status_code==200
    assert next(r for r in c.get('/api/motus').json()['items'] if r['id']=='wave')['name']=='Draft wave'
    assert len(c.get('/api/admin/motus/wave/revisions').json()['items'])==3

def test_legacy_import_is_idempotent_and_preserves_source_files(tmp_path):
    root=tmp_path/'data';p=root/'projects'/str(uuid.uuid4());p.mkdir(parents=True)
    record={'id':p.name,'name':'Existing spectrum','status':'ready','created_at':1}
    (p/'project.json').write_text(json.dumps(record))
    store=Store('sqlite:///'+str(tmp_path/'migration.sqlite'));owner=store.seed_owner(['owner@example.com'])
    store.import_legacy(root,owner);store.import_legacy(root,owner)
    assert len(store.list_projects())==1
    assert store.get_project(p.name)['owner_id']==owner
    assert (p/'project.json').exists()
    assert (root/'backups/before-postgres/projects'/p.name/'project.json').exists()


def test_public_landing_private_studio_and_secure_logout(studio):
    app,c,m=studio
    assert c.get('/').status_code==200
    assert 'A photo that' in c.get('/').text
    assert c.get('/studio',follow_redirects=False).headers['location']=='/login'
    # Exercise the production cookie prefix and a real HTTPS Origin.
    a=app.state.accounts;a.secure=True;a.cookie='__Host-spectramotus';a.app_url='https://testserver'
    with TestClient(app,base_url='https://testserver') as browser:
        login(browser,m)
        raw=browser.cookies.get(a.cookie)
        assert browser.get('/',follow_redirects=False).headers['location']=='/studio'
        page=browser.get('/studio')
        assert page.headers['referrer-policy']=='same-origin'
        assert browser.get('/auth/verify').headers['referrer-policy']=='no-referrer'
        assert browser.post('/logout',headers={'Origin':'https://evil.example'}).status_code==403
        assert a.session(raw)
        response=browser.post('/logout',headers={'Origin':'https://testserver'},follow_redirects=False)
        assert response.status_code==303 and response.headers['location']=='/'
        cookie=next(v for v in response.headers.get_list('set-cookie') if a.cookie in v)
        assert 'Secure' in cookie and 'HttpOnly' in cookie and 'Max-Age=0' in cookie
        assert not a.session(raw)
        assert browser.get('/api/me').status_code==401
        assert 'A photo that' in browser.get('/').text
        assert browser.post('/logout',headers={'Origin':'https://testserver'}).status_code==200


def test_email_design_selection_and_single_use_invitation(studio):
    app,c,m=studio
    login(c,m)
    assert m[-1][2]=='signin'
    ident=c.post('/api/admin/users',data={'email':'new@example.com','role':'beta'}).json()['id']
    assert c.post('/api/admin/users/'+ident+'/send').status_code==200
    assert m[-1][2]=='invite'
    raw=token(m)
    newcomer=TestClient(app)
    assert newcomer.post('/auth/consume',data={'token':raw}).status_code==200
    assert newcomer.post('/auth/consume',data={'token':raw}).status_code==400
    newcomer.post('/auth/request',data={'email':'new@example.com'})
    assert m[-1][2]=='signin'


def test_branded_resend_payload(monkeypatch):
    import base64
    from spectramotus.studio.accounts import ResendMailer
    monkeypatch.setenv('RESEND_API_KEY','test-only')
    monkeypatch.setenv('RESEND_FROM_EMAIL','hello@example.com')
    payloads=[]
    def capture(url,**kwargs):
        payloads.append(kwargs['json'])
        return type('Response',(),{'status_code':200})()
    monkeypatch.setattr('spectramotus.studio.accounts.httpx.post',capture)
    url='https://example.com/auth/verify#token=test&value="escaped"'
    for kind in ('invite','signin'):
        ResendMailer()('recipient@example.com',url,kind=kind)
    invite,signin=payloads
    assert invite['subject']=='You’re invited to Spectramotus'
    assert 'Accept your invitation' in invite['html']
    assert 'Sign in to your studio' in signin['html']
    for payload in payloads:
        assert payload['to']==['recipient@example.com']
        assert url in payload['text']
        assert '&amp;value=&quot;escaped&quot;' in payload['html']
        assert '{{action_url}}' not in payload['html'] and 'preview-only' not in payload['html']
        assert '15 minutes' in payload['text']
    assert 'attachments' not in signin
    attachment=invite['attachments'][0]
    assert 'cid:'+attachment['content_id'] in invite['html']
    assert base64.b64decode(attachment['content']).startswith(b'\x89PNG\r\n\x1a\n')
