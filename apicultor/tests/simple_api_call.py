#!/usr/bin/python3

import http.client
conn = http.client.HTTPConnection("apicultor:5000")

payload = "filePath=%2Fmedia%2Fmnt%2Ffiles%2Ftest2222-file.pdf"

headers = {
    'cache-control': "no-cache",
    }

conn.request("GET", "/search/mir/samples/HFC/greaterthan/40000/5", payload, headers)

res = conn.getresponse()
data = res.read()

print((data.decode("utf-8")))
