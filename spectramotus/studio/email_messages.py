"""Branded transactional emails with a plain-text alternative."""
import base64
from html import escape
from pathlib import Path

TEMPLATES = Path(__file__).with_name('email_templates')


def email_message(url, kind='signin'):
    if kind not in {'invite', 'signin'}:
        raise ValueError('Unknown email kind')
    invite = kind == 'invite'
    subject = 'You’re invited to Spectramotus' if invite else 'Your Spectramotus sign-in link'
    intro = ('You’re invited to Spectramotus. Bring your photos to life, choose their movements, '
             'and try them in your studio.' if invite else
             'Your studio is ready. Use the link below to sign in to Spectramotus.')
    action = 'Accept your invitation' if invite else 'Sign in to your studio'
    notice = ('Not expecting an invitation? You can safely ignore this email.' if invite else
              'If you didn’t request this link, you can safely ignore this email.')
    message = {
        'subject': subject,
        'html': (TEMPLATES / f'{kind}.html').read_text().replace('{{action_url}}', escape(url, quote=True)),
        'text': f'{intro}\n\n{action}:\n{url}\n\nThis link works once and expires in 15 minutes.\n\n{notice}\n\nSpectramotus · Images in motion.',
    }
    if invite:
        message['attachments'] = [{
            'filename': 'portrait.png', 'content_type': 'image/png',
            'content_id': 'spectramotus-portrait',
            'content': base64.b64encode((TEMPLATES / 'portrait.png').read_bytes()).decode('ascii'),
        }]
    return message
