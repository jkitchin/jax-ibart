"""mkdocs hook: fix relative links inside rendered notebooks.

mkdocs-jupyter renders ``examples/NN.ipynb`` at ``examples/NN/`` and does not
rewrite links, so ``02_catalysis.ipynb`` and ``../references.md`` (correct on
GitHub) would resolve one level too deep on the built site.
"""

import re

_LINK = re.compile(r'href="(?!https?:|#|/)([^"#]+?)\.(ipynb|md)(#[^"]*)?"')


def on_page_content(html, page, config, files):
    if not page.file.src_uri.endswith(".ipynb"):
        return html

    def fix(m):
        path, anchor = m.group(1), m.group(3) or ""
        return f'href="../{path}/{anchor}"'

    return _LINK.sub(fix, html)
