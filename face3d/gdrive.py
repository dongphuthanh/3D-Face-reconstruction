"""Google Drive downloader that handles the current confirmation flow.

FFHQ's official download_ffhq.py predates a change in how Drive gates large
files: it now serves an HTML interstitial containing a form that must be
submitted to drive.usercontent.google.com. The old script's confirm-token logic
silently receives that HTML, writes it to disk, and fails with "Incorrect file
size" — which reads as a corrupt download rather than an auth flow change.
"""

import hashlib
import pathlib
import re

import requests

CHUNK = 1 << 20


def download(file_id, dest, expected_size=None, expected_md5=None, session=None,
             progress=None):
    """Fetch a Drive file by id. Returns the destination path."""
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()

    r = s.get("https://drive.google.com/uc", params={"id": file_id, "export": "download"},
              stream=True, timeout=60)
    ctype = r.headers.get("content-type", "")
    if "text/html" in ctype:
        html = r.text
        action = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', html)
        if not action:
            raise IOError(f"Drive returned HTML with no download form for {file_id}. "
                          f"Usually a quota or permission problem.")
        fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', html))
        r = s.get(action.group(1).replace("&amp;", "&"), params=fields,
                  stream=True, timeout=60)
    r.raise_for_status()

    h = hashlib.md5()
    n = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(CHUNK):
            if not chunk:
                continue
            f.write(chunk); h.update(chunk); n += len(chunk)
            if progress:
                progress(n, expected_size)

    if expected_size is not None and n != expected_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"size mismatch for {dest.name}: got {n}, expected {expected_size}")
    if expected_md5 is not None and h.hexdigest() != expected_md5:
        tmp.unlink(missing_ok=True)
        raise IOError(f"md5 mismatch for {dest.name}")
    tmp.replace(dest)
    return dest
