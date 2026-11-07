"""Pinned public source bytes; existing files are checked rather than replaced."""
from pathlib import Path
import hashlib
import os
import tempfile
import time
import urllib.request

PRIMARY_SHA = 'd0da254163384f6e1276f6fcb1cd3ca8cf4bd515a7e8817d9766b2f2bd7c9eeb'
EXTERNAL_SHA = '6afba982acaff9bb7bee28c10c0df7af5337819b49164043e14f2a318375a566'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
SOURCES = {
 'primary': ('data/raw/indian_transport/grievance.csv',
   'https://raw.githubusercontent.com/sohampoddar26/indian-transport-snam/main/grievance.csv', PRIMARY_SHA),
 'external': ('data/raw/dabiri_traffic/1_TrainingSet_3Class.csv',
   'https://raw.githubusercontent.com/sinadabiri/Tweet-Classification-Deep-Learning-Traffic/4cb11943af33057f09540160f4bd0c01a33cbecd/Data%20Collection/1_TrainingSet_3Class.csv', EXTERNAL_SHA),
 'encoder': ('assets/pretrained/encoder/onnx/model.onnx',
   f'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{REVISION}/onnx/model.onnx',
   '6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452'),
 'tokenizer': ('assets/pretrained/encoder/tokenizer.json',
   f'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{REVISION}/tokenizer.json',
   'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037')}


def obtain(root: Path, item: tuple, logger, download=True) -> dict:
    from ..runtime.artifacts import safe_path, file_hash
    relative, url, expected = item
    path = safe_path(root, relative)
    if path.exists():
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError('Existing source checksum differs; it was preserved: ' + relative)
        logger.info('Source verified: %s', relative)
        return {'path': relative, 'url': url, 'sha256': expected, 'bytes': path.stat().st_size, 'downloaded': False}
    if not download:
        raise FileNotFoundError('Download disabled; place the verified file at ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if 'huggingface.co/' in url:
        import shutil
        relative_model='onnx/model.onnx' if relative.endswith('model.onnx') else 'tokenizer.json'
        cached=Path.home()/'.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots'/REVISION/relative_model
        if cached.is_file() and file_hash(cached)==expected:
            shutil.copyfile(cached,path)
            logger.info('Reused verified local Hugging Face cache: %s',relative)
            return {'path':relative,'url':url,'sha256':expected,'bytes':path.stat().st_size,'downloaded':False}
    failure = None
    for attempt in range(2):
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.download_')
        try:
            logger.info('Downloading %s (attempt %d/2).', relative, attempt+1)
            logger.info('Public source URL: %s',url)
            request = urllib.request.Request(url, headers={'User-Agent': 'proposed-transport-topics/0.2'})
            digest, size, announced = hashlib.sha256(), 0, time.monotonic()
            with os.fdopen(descriptor, 'wb') as target, urllib.request.urlopen(request, timeout=60) as source:
                while True:
                    block = source.read(1024*1024)
                    if not block: break
                    target.write(block); digest.update(block); size += len(block)
                    if time.monotonic()-announced > 10:
                        logger.info('Downloading %s: %.1f MB received.', relative, size/1024**2)
                        announced = time.monotonic()
                target.flush(); os.fsync(target.fileno())
            if digest.hexdigest() != expected:
                raise ValueError('Downloaded checksum mismatch: ' + relative)
            os.replace(temporary, path)
            logger.info('Downloaded and verified %s (%.1f MB).', relative, size/1024**2)
            return {'path':relative, 'url':url, 'sha256':expected, 'bytes':size, 'downloaded':True}
        except Exception as error:
            failure = error
            try: os.close(descriptor)
            except OSError: pass
            logger.warning('Source acquisition failed: %s', str(error))
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    raise RuntimeError('Could not acquire %s. Existing files were preserved. Manually download %s to %s. Last error: %s'
                       % (relative, url, path, failure))
