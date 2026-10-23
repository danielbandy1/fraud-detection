"""Anonymous public download; unchanged archive + receipt. AUTHORED BY ORION."""
from pathlib import Path
import urllib.request,hashlib,json,datetime
ROOT=Path(__file__).resolve().parent
URL='https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1'
if __name__=='__main__':
 p=ROOT/'data/raw/paysim_source.zip'
 if not p.exists():
  with urllib.request.urlopen(URL,timeout=60) as r,p.open('xb') as f:
   while block:=r.read(1024*1024):f.write(block)
  status=r.status
 else:status='cached'
 h=hashlib.sha256()
 with p.open('rb') as f:
  while block:=f.read(1024*1024):h.update(block)
 receipt={'authored_by':'ORION','url':URL,'downloaded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 (ROOT/'data/raw/paysim_download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
