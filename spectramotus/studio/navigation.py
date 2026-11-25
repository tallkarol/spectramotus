"""Render the shared navigation for hosted Studio pages."""
from html import escape

from .packs import STATIC


NAV_STYLESHEET = '/studio-nav.css'
ADMIN_LINKS = (
    ('/admin', 'Motus library'),
    ('/admin/users', 'People'),
    ('/admin/pi', 'Pi kit'),
)


def render_page(request, filename):
    html = (STATIC / filename).read_text()
    user = request.state.user
    path = request.url.path
    if user:
        home = '/studio'
        links = [('/welcome', 'About Spectramotus')]
        if user['role'] in {'owner', 'admin'}:
            links.extend(ADMIN_LINKS)
        account = f'''<details class="studio-nav__account" name="studio-nav">
            <summary class="studio-nav__action">Your account</summary>
            <div class="studio-nav__panel">
                <span class="studio-nav__email">{escape(user['email'])}</span>
                <form action="/logout" method="post">
                    <button class="studio-nav__sign-out" type="submit">Sign out</button>
                </form>
            </div>
        </details>'''
    else:
        home = '/'
        how_it_works = '#how-it-works' if path in {'/', '/welcome'} else '/#how-it-works'
        links = [(how_it_works, 'How it works')]
        account = f'<a class="studio-nav__action" href="/login"{current_page(path, "/login")}>Sign in</a>'

    items = '\n'.join(
        f'<a class="studio-nav__link" href="{href}"{current_page(path, href)}>{label}</a>'
        for href, label in links
    )
    nav = f'''<header class="studio-nav">
    <a class="studio-nav__wordmark" href="{home}"{current_page(path, home)}>spectramotus<span>.</span></a>
    <nav class="studio-nav__actions" aria-label="Main navigation">
        <div class="studio-nav__links">{items}</div>
        <details class="studio-nav__menu" name="studio-nav">
            <summary class="studio-nav__toggle" aria-label="Menu"><span aria-hidden="true"></span></summary>
            <div class="studio-nav__panel">{items}</div>
        </details>
        {account}
    </nav>
</header>'''
    return html.replace('<!-- studio-nav -->', nav, 1).replace(
        '</head>', f'<link rel="stylesheet" href="{NAV_STYLESHEET}"></head>', 1
    )


def current_page(path, href):
    return ' aria-current="page"' if path == href else ''
