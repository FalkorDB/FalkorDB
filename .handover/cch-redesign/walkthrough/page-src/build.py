"""Assemble index.html from page-src/ and example.json (standard library only: python3 -I build.py)."""
import json
import pathlib

SRC = pathlib.Path(__file__).resolve().parent
WT = SRC.parent


def main():
    obj = json.loads((WT / "example.json").read_text(encoding="utf-8"))
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/").replace("<!--", "<\\u0021--")
    css = (SRC / "style.css").read_text(encoding="utf-8")
    body = (SRC / "body.html").read_text(encoding="utf-8")
    app = "\n".join((SRC / f"app{i}.js").read_text(encoding="utf-8") for i in (1, 2, 3, 4, 5))
    script = "(function () {\n" + app + "\n})();\n"
    (SRC / "app.bundle.js").write_text(script, encoding="utf-8")
    out = (
        "<title>CCH GraphBLAS Walkthrough</title>\n"
        '<meta name="claude:viewport-fit-sides" content="cover">\n'
        "<style>\n" + css + "</style>\n"
        + body
        + '<script type="application/json" id="data">' + data + "</script>\n"
        + "<script>\n" + script + "</script>\n"
    )
    (WT / "index.html").write_text(out, encoding="utf-8")
    print("index.html", len(out.encode("utf-8")), "bytes")


if __name__ == "__main__":
    main()
