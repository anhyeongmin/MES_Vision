"""Download a pinned Meta SAM 2.1 model for offline annotation only."""
from pathlib import Path
import hashlib,json,urllib.request
from huggingface_hub import HfApi,hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
REPO='facebook/sam2.1-hiera-small'
REVISION='ee5bba1d82bb8749febdf90f45e84b687142ba03'
FILES=['config.json','model.safetensors','preprocessor_config.json','processor_config.json','README.md']

def main():
    output=ROOT/'models/sam2.1-hiera-small';output.mkdir(parents=True,exist_ok=True)
    info=HfApi().model_info(REPO,revision=REVISION,files_metadata=True)
    entries={item.rfilename:item for item in info.siblings}
    records={}
    for name in FILES:
        path=Path(hf_hub_download(REPO,name,revision=REVISION,local_dir=output))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        lfs=entries[name].lfs
        if lfs: assert digest==lfs.sha256, 'Publisher checksum mismatch'
        records[name]={'sha256':digest,'bytes':path.stat().st_size}
    license_dir=ROOT/'licenses/sam2';license_dir.mkdir(parents=True,exist_ok=True)
    license_url='https://raw.githubusercontent.com/facebookresearch/sam2/main/LICENSE'
    license_data=urllib.request.urlopen(license_url).read()
    assert b'Apache License' in license_data
    (license_dir/'LICENSE').write_bytes(license_data)
    manifest={'repository':REPO,'revision':REVISION,'license':'Apache-2.0','license_source':license_url,
              'license_sha256':hashlib.sha256(license_data).hexdigest(),'purpose':'annotation assistance only',
              'files':records,'runtime':'installed transformers; local_files_only; no remote code'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('SAM 2.1 Small pinned files verified; no production model registration.')

if __name__=='__main__':main()
