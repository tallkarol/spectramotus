"""Small relational store shared by PostgreSQL production and SQLite test fixtures."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid
from sqlalchemy import (create_engine, MetaData, Table, Column, String, Integer, BigInteger,
                        JSON, ForeignKey, select, insert, update, delete, text)

meta=MetaData()
users=Table('users',meta,Column('id',String(36),primary_key=True),Column('email',String(254),unique=True,nullable=False),Column('role',String(16),nullable=False),Column('status',String(16),nullable=False),Column('invite_expires_at',BigInteger),Column('created_at',BigInteger,nullable=False))
emails=Table('user_emails',meta,Column('email',String(254),primary_key=True),Column('user_id',String(36),ForeignKey('users.id'),nullable=False))
links=Table('sign_in_links',meta,Column('hash',String(64),primary_key=True),Column('user_id',String(36),ForeignKey('users.id'),nullable=False),Column('email',String(254),nullable=False),Column('expires_at',BigInteger,nullable=False),Column('used_at',BigInteger))
sessions=Table('sessions',meta,Column('hash',String(64),primary_key=True),Column('user_id',String(36),ForeignKey('users.id'),nullable=False),Column('expires_at',BigInteger,nullable=False))
limits=Table('auth_rate_limits',meta,Column('key',String(64),primary_key=True),Column('window',BigInteger,nullable=False),Column('count',Integer,nullable=False))
spectra=Table('spectra',meta,Column('id',String(36),primary_key=True),Column('owner_id',String(36),ForeignKey('users.id'),nullable=False,index=True),Column('data',JSON,nullable=False))
motus=Table('motus',meta,Column('id',String(80),primary_key=True),Column('data',JSON,nullable=False),Column('published_revision',Integer))
revisions=Table('motus_revisions',meta,Column('motus_id',String(80),ForeignKey('motus.id'),primary_key=True),Column('revision',Integer,primary_key=True),Column('data',JSON,nullable=False))
migrations=Table('schema_migrations',meta,Column('version',String(80),primary_key=True),Column('applied_at',BigInteger,nullable=False))
audit=Table('audit_events',meta,Column('id',String(36),primary_key=True),Column('actor_id',String(36)),Column('action',String(60),nullable=False),Column('target_id',String(80)),Column('created_at',BigInteger,nullable=False))

def fingerprint(token):return hashlib.sha256(token.encode()).hexdigest()

class Store:
    def __init__(self,url):
        if url.startswith(('postgres://','postgresql://')):url='postgresql+psycopg://'+url.split('://',1)[1]
        self.engine=create_engine(url,pool_pre_ping=True,hide_parameters=True,**({'connect_args':{'check_same_thread':False,'timeout':30}} if url.startswith('sqlite:') else {}))
        # Version 1 is additive. PostgreSQL serializes simultaneous startup migrations.
        with self.engine.begin() as c:
            if self.engine.dialect.name=='postgresql':c.execute(text('SELECT pg_advisory_xact_lock(784201)'))
            meta.create_all(c)
            if not c.execute(select(migrations).where(migrations.c.version=='001_accounts_motus_spectra')).first():
                c.execute(insert(migrations).values(version='001_accounts_motus_spectra',applied_at=int(time.time())))

    def event(self,c,actor,action,target):
        c.execute(insert(audit).values(id=str(uuid.uuid4()),actor_id=actor,action=action,target_id=target,created_at=int(time.time())))

    def seed_owner(self,addresses):
        with self.engine.begin() as c:
            row=c.execute(select(users).where(users.c.email==addresses[0])).mappings().first()
            if row:
                if row['role']!='owner':raise RuntimeError('Configured owner already exists with a different role.')
                ident=row['id']
            else:
                ident=str(uuid.uuid4());c.execute(insert(users).values(id=ident,email=addresses[0],role='owner',status='active',created_at=int(time.time())))
            for address in addresses:
                existing=c.execute(select(emails).where(emails.c.email==address)).mappings().first()
                if existing and existing['user_id']!=ident:raise RuntimeError('Owner email belongs to another account.')
                if not existing:c.execute(insert(emails).values(email=address,user_id=ident))
        return ident

    def by_email(self,address,c=None):
        if c is None:
            with self.engine.connect() as conn:return self.by_email(address,conn)
        row=c.execute(select(users).join(emails,emails.c.user_id==users.c.id).where(emails.c.email==address)).mappings().first()
        return dict(row) if row else None

    def list_projects(self):
        with self.engine.connect() as c:return [dict(r['data'],owner_id=r['owner_id']) for r in c.execute(select(spectra)).mappings()]

    def get_project(self,ident):
        with self.engine.connect() as c:
            r=c.execute(select(spectra).where(spectra.c.id==ident)).mappings().first()
            if not r:raise FileNotFoundError(ident)
            return dict(r['data'],owner_id=r['owner_id'])

    def save_project(self,record,default_owner):
        with self.engine.begin() as c:
            exists=c.execute(select(spectra.c.id).where(spectra.c.id==record['id'])).first()
            values=dict(owner_id=record.get('owner_id',default_owner),data=record)
            if exists:c.execute(update(spectra).where(spectra.c.id==record['id']).values(**values))
            else:c.execute(insert(spectra).values(id=record['id'],**values))

    def import_legacy(self,root,owner_id):
        """One-time metadata import. Files remain intact as a rollback source."""
        root=Path(root)
        with self.engine.begin() as c:
            if self.engine.dialect.name=='postgresql':c.execute(text('SELECT pg_advisory_xact_lock(784202)'))
            if c.execute(select(migrations).where(migrations.c.version=='002_legacy_json_import')).first():return
            backup=root/'backups'/'before-postgres';backup.mkdir(parents=True,exist_ok=True)
            for folder in ('projects','motus'):
                for path in (root/folder).glob('*/project.json' if folder=='projects' else '*.json'):
                    dest=backup/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():shutil.copyfile(path,dest)
                    record=json.loads(path.read_text())
                    table=spectra if folder=='projects' else motus
                    if c.execute(select(table.c.id).where(table.c.id==record['id'])).first():continue
                    if folder=='projects':c.execute(insert(spectra).values(id=record['id'],owner_id=owner_id,data=dict(record,owner_id=owner_id)))
                    else:
                        record.setdefault('publication','published' if record['enabled'] else 'disabled')
                        c.execute(insert(motus).values(id=record['id'],data=record,published_revision=record['revision'] if record['enabled'] else None))
                        c.execute(insert(revisions).values(motus_id=record['id'],revision=record['revision'],data=record))
            c.execute(insert(migrations).values(version='002_legacy_json_import',applied_at=int(time.time())))
