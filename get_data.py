"""Only needed if data/ is empty. Downloads the UCI Bank Marketing data (bank-additional-full.csv) from a public
GitHub mirror pinned to one commit, and checks it against a fixed SHA-256 so the numbers always reproduce.
Original source: https://archive.ics.uci.edu/dataset/222/bank+marketing"""
import gzip, hashlib, urllib.request, pathlib
URL = ("https://raw.githubusercontent.com/ahanagangopadhyay/improving-bank-telemarketing-campaigns/"
       "261a9c65cf4024c74e5b17aaa2e178749c682f15/data/bank-additional-full.csv")
SHA = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
raw = urllib.request.urlopen(URL).read()
assert hashlib.sha256(raw).hexdigest() == SHA, "downloaded file does not match the expected checksum"
pathlib.Path("data").mkdir(exist_ok=True)
gzip.open("data/bank-additional-full.csv.gz", "wb").write(raw)
print("saved data/bank-additional-full.csv.gz")
