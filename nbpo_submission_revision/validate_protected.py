#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'protected_manifest.json').read_text())
changed=[name for name,h in manifest.items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=h]
if changed: print('PROTECTED FILE CHANGES:',*changed,sep='\n');sys.exit(1)
print('Protected prose, reviews, theory, existing results and template structure unchanged.')
