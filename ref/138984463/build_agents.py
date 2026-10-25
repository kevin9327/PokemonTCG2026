"""Reproduce public MIT policies with byte-pinned inputs and explicit credits.

The byte deltas and unified diff are Akmal Xodarev's public notebook work.
The checkpoint/runtime are Dipam Chakraborty's, ported by Cristobal Tudela.
Our contribution is the controlled comparison, failure audit and packaging.
"""
import base64
import hashlib
import json
import lzma
import re
import shutil
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def apply_text_patch(files, patch):
    files = dict(files)
    for chunk in re.split(r'^--- ', patch, flags=re.M)[1:]:
        lines = chunk.splitlines(keepends=True)
        path = lines[1][len('+++ b/'):].strip()
        assert not Path(path).is_absolute() and '..' not in Path(path).parts
        text = files.get(path, '')
        for hunk in re.split(r'^@@[^\n]*@@\n', ''.join(lines[2:]), flags=re.M)[1:]:
            old = ''.join(line[1:] for line in hunk.splitlines(keepends=True) if line[0] in ' -')
            new = ''.join(line[1:] for line in hunk.splitlines(keepends=True) if line[0] in ' +')
            if old:
                assert text.count(old) == 1, path
                text = text.replace(old, new)
            else:
                text += new
        files[path] = text
    return files


def apply_byte_delta(payload, sources):
    header, literals = lzma.decompress(base64.b64decode(payload)).split(b'\n', 1)
    source, result, position = b''.join(sources), bytearray(), 0
    for op in json.loads(header):
        if op[0] == 0:
            result += literals[position:position+op[1]]
            position += op[1]
        else:
            assert op[0] == 1 and 0 <= op[1] <= len(source)
            result += source[op[1]:op[1]+op[2]]
    return bytes(result)


def write_files(folder, files):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        target = folder/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data if isinstance(data, bytes) else data.encode('utf-8'))
    return folder


def copy_official_runtime(folder, sdk, card_csv):
    shutil.copytree(Path(sdk)/'cg', folder/'cg', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    (folder/'data').mkdir(exist_ok=True)
    for target in (folder/'EN_Card_Data.csv', folder/'data'/'EN_Card_Data.csv'):
        shutil.copyfile(card_csv, target)


def build_variants(base_folder, sdk, card_csv, assets, scratch, previous_source):
    """Build outside public notebook Output: official runtime is restricted input."""
    base_folder, scratch = Path(base_folder), Path(scratch)
    originals = {}
    for name, expected in assets['DIPAM_SHA'].items():
        data = (base_folder/name).read_bytes()
        if name in assets['ORIGINAL_SOURCES']:
            data = assets['ORIGINAL_SOURCES'][name].encode('utf-8')
        assert sha(data) == expected, name
        originals[name] = data
    license_text = (base_folder/'LICENSE').read_bytes()
    files = dict(originals)
    for name, source in apply_text_patch(
        {name: value.decode() for name, value in originals.items() if name.endswith('.py')},
        assets['PATCH']).items():
        files[name] = source.encode()
    files['model.pt'] = apply_byte_delta(assets['MODEL_DELTA'], [originals['model.pt']])
    files['data/embed_pca.npz'] = apply_byte_delta(
        assets['NPZ_DELTA'], [originals['data/embed_pca.npz'], files['model.pt']])
    files['r2e_manifest.json'] = assets['R2E_MANIFEST'].encode()
    # The author's exact submitted-file hashes verify the reconstruction.
    expected = assets['PUBLIC_V3_MANIFEST']
    for name, data in files.items():
        if name == 'deck.csv':
            expected_hash = 'multiset:' + sha('\n'.join(sorted(data.decode().split())).encode())
            assert expected_hash == expected[name]
        else:
            assert sha(data) == expected[name], name
    files['LICENSE'] = license_text
    files['THIRD_PARTY_NOTICES.txt'] = (
        'Policy/checkpoint: Dipam Chakraborty, MIT, pinned commit '+assets['DIPAM_COMMIT']+'\n'
        'Compatibility port: Cristobal Tudela, MIT, Kaggle public baseline dataset.\n'
        'BO3 seat/tracker fixes and new-card embeddings: Akmal Xodarev (busyaprime),\n'
        'https://www.kaggle.com/code/busyaprime/ptcg-bo3-agent-one-line-seat-fix-lb-1132-4\n'
        'This is a credited reproduction, not a claim that our team trained the model.\n'
        'Our work: byte verification, previous-agent control, stronger opponents,\n'
        'four seat/order state audit, independent test and submission verification.\n').encode()
    extended = write_files(scratch/'newcard_extended', files)
    copy_official_runtime(extended, sdk, card_csv)

    port_files = {name: (base_folder/name).read_bytes() for name in assets['DIPAM_SHA']}
    port_files['LICENSE'] = license_text
    public_port = write_files(scratch/'public_port', port_files)
    copy_official_runtime(public_port, sdk, card_csv)
    fixed_files = dict(port_files)
    encoder = fixed_files['ptcg/rl/encoder.py'].decode()
    old = '    g[GlobF.ACTOR_SEAT] = float(seat)\n'
    new = '    g[GlobF.ACTOR_SEAT] = 0.0 if first == -1 else float(first != seat)\n'
    assert encoder.count(old) == 1
    fixed_files['ptcg/rl/encoder.py'] = encoder.replace(old, new).encode()
    # Main's BO3 reset and first/second answer are the public author's fixes.
    fixed_files['main.py'] = files['main.py']
    fixed_files['THIRD_PARTY_NOTICES.txt'] = files['THIRD_PARTY_NOTICES.txt']
    bo3_fixed = write_files(scratch/'bo3_fixed', fixed_files)
    copy_official_runtime(bo3_fixed, sdk, card_csv)

    previous = write_files(scratch/'previous_rule', {'main.py': previous_source})
    # Old submitted source is pure stdlib, and its own deck entry point is used.
    from ast import parse, Assign, Name, literal_eval
    ns = {}
    exec(compile(previous_source, 'previous_main.py', 'exec'), ns)
    deck = ns['agent']({'select': None})
    (previous/'deck.csv').write_text(''.join(f'{i}\n' for i in deck), encoding='utf-8')
    variants = dict(previous_rule=previous, public_port=public_port,
                    bo3_fixed=bo3_fixed, newcard_extended=extended)
    for name, item in assets['RULE_OPPONENTS'].items():
        folder = write_files(scratch/name, {'main.py':item['source'],
                             'deck.csv': ''.join(f'{i}\n' for i in item['deck'])})
        shutil.copytree(Path(sdk)/'cg', folder/'cg', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
        variants[name] = folder
    return {name: str(path.resolve()) for name, path in variants.items()}
