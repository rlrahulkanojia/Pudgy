import os, sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
load_dotenv("/workspace/Pudgy/.env")
c=BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"]).get_container_client("pudgy")
prefixes=sys.argv[1:]
blobs=[b for p in prefixes for b in c.list_blobs(name_starts_with=p)]
def get(b):
    p=os.path.join("/workspace/Data",b.name)
    if os.path.exists(p) and os.path.getsize(p)==b.size: return 0
    os.makedirs(os.path.dirname(p),exist_ok=True)
    with open(p,"wb") as f: c.download_blob(b.name,max_concurrency=4).readinto(f)
    return b.size
with ThreadPoolExecutor(16) as ex: n=sum(ex.map(get,blobs))
print(f"{len(blobs)} blobs, {n/1e9:.2f} GB downloaded")
