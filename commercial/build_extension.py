"""Build an isolated Chrome Web Store ZIP. Never edits chrome-extension/."""
import argparse
import json
import zipfile
from pathlib import Path
from urllib.parse import urlparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', required=True, help='Final HTTPS origin, e.g. https://manga.example.com')
    args = parser.parse_args()
    url = urlparse(args.origin)
    if (url.scheme != 'https' or not url.hostname or url.username or url.password or url.port
            or url.path not in {'', '/'} or url.query or url.fragment or url.hostname in {'localhost', '127.0.0.1'}
            or url.hostname.endswith('.invalid')):
        parser.error('Use a real public HTTPS origin without a port, path, credentials, query or fragment')
    origin = f'https://{url.hostname}'
    root = Path(__file__).parent
    output = root / 'extension-build'
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(output / 'mangatranslator-cloud.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in (root / 'extension').iterdir():
            if source.name == 'manifest.json':
                manifest = json.loads(source.read_text(encoding='utf8'))
                manifest['host_permissions'] = [origin + '/*']
                archive.writestr(source.name, json.dumps(manifest, ensure_ascii=False, indent=2))
            elif source.name == 'config.js':
                archive.writestr(source.name, 'const API_ORIGIN = ' + json.dumps(origin) + ';\n')
            else:
                archive.write(source, source.name)
    print(output / 'mangatranslator-cloud.zip')

if __name__ == '__main__':
    main()
