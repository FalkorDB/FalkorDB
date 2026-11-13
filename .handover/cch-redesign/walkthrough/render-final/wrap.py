"""Wrap index.html in the publish skeleton (light: no theme attribute; dark: data-theme="dark"). python3 -I wrap.py"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PAGE = (HERE.parent / "index.html").read_text(encoding="utf-8")
RESET = (
    "<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);"
    "padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,-apple-system,sans-serif}"
    "img{max-width:100%}[hidden]{display:none!important}</style>"
)


def wrap(attr):
    return ('<!doctype html><html' + attr + '><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            + RESET + '</head><body>' + PAGE + '</body></html>')


for name, attr in (("light", ""), ("dark", ' data-theme="dark"'), ("forcelight", ' data-theme="light"')):
    (HERE / (name + ".html")).write_text(wrap(attr), encoding="utf-8")
print("ok")
