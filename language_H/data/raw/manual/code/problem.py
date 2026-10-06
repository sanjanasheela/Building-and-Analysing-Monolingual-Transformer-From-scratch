import requests

url = "https://www.eenadu.net/telugu-news/movies/rajinikanth-reveals-why-he-is-hesitant-to-speak/0210/126110901"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

try:
    r = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    print("STATUS:", r.status_code)
    print("FINAL URL:", r.url)
    print("CONTENT TYPE:", r.headers.get("content-type"))
    print("CONTENT LENGTH:", len(r.content))
    print()
    print(r.text[:1000])

    with open(
        "eenadu_debug.html",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(r.text)

except Exception as e:
    print("ERROR:", repr(e))