import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

urls = [
    "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/",
    "https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/"
]
out_dir = "syzygy"

os.makedirs(out_dir, exist_ok=True)

files_to_download = []

for url in urls:
    print(f"Fetching index for {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    files = re.findall(r'href="([^"]+\.rtb[wz])"', html)
    for f in files:
        files_to_download.append((url + f, f))

print(f"Found {len(files_to_download)} tablebase files.")

def download_file(item):
    file_url, filename = item
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        return
    try:
        req = urllib.request.Request(file_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Failed {filename}: {e}")

print("Starting download...")
with ThreadPoolExecutor(max_workers=16) as executor:
    executor.map(download_file, files_to_download)
    
print("Download complete.")
