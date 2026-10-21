"""Dependency-free binding, validation and rendering of grading inputs."""

from array import array
import json
import math
from pathlib import Path
import struct
import sys
import zlib

from mri_root_impact.contracts import load_document, manifest_digest, validate_bundle
from mri_root_impact.spinenet_assets import file_sha256, verify_assets
from mri_root_impact.spinenet_runner import validate_worker_outputs

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ('L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1')
SCALE = 'spinenet_v2_herniation_binary'
SHAPE = [9, 112, 224]
REASON = 'Native binary prediction; anatomical numbering and T2 contrast unreviewed. Side, size, morphology and root impact not assessed.'


def code_hashes():
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted((ROOT / 'mri_disc').glob('*.py'))}


def write(path, document):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def input_hashes(study_path, spine_run):
    study = load_document(study_path)
    paths = [Path(study_path)] + [Path(spine_run) / n for n in
            ('native.json', 'spatial.json', 'execution.json', 'localization.json', 'run.json')]
    result = {str(p.resolve()): file_sha256(p) for p in paths}
    for series in study['series']:
        for image in series['images']:
            if image['sha256'] is None or file_sha256(image['path']) != image['sha256']:
                raise ValueError('Disc source image SHA is missing or changed')
            result[str(Path(image['path']).resolve())] = image['sha256']
    return result


def png_from_input(raw):
    """Exact central (5th of 9) model-input plane; no lesion annotation."""
    if len(raw) != 9 * 112 * 224 * 8:
        raise ValueError('Disc crop must be little-endian float64, 9x112x224')
    values = array('d'); values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Disc model inputs must be finite in [0,1]')
    plane = bytes(int(v * 255) for v in values[4 * 112 * 224:5 * 112 * 224])
    scanlines = b''.join(b'\0' + plane[y * 224:(y+1) * 224] for y in range(112))
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 224, 112, 8, 0, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))


def source_samples(detections, geometry, level):
    ordered = sorted(detections, key=lambda d: sum(p[1] for p in d['average_polygon']) / 4, reverse=True)
    def median(values):
        values = sorted(values); n = len(values)
        return values[n//2] if n % 2 else (values[n//2-1] + values[n//2]) / 2
    for a, b in zip(ordered, ordered[1:]):
        if b['predicted_label'] + '-' + a['predicted_label'] != level:
            continue
        center = round((round(median(a['slice_nos'])) + round(median(b['slice_nos']))) / 2)
        slices = geometry['slices']
        samples = [slices[i]['image_ref'] if 0 <= i < len(slices) else None for i in range(center-4, center+5)]
        return samples
    return None


def build_result(study, localization, document, assets):
    assessments = []
    for row in document['predictions']:
        level = row['native_level']
        refs = list({manifest_digest(ref): ref for ref in row['source_samples'] if ref is not None}.values())
        upstream = [{'result_id': localization['result_id'], 'assessment_id': item['assessment_id']}
                    for item in localization['assessments'] if item['observation'] is not None
                    and item['observation']['label'] in level.split('-')]
        value = row['native_class']
        assessments.append({'assessment_id': 'disc-' + level,
                            'target': {'level': level, 'side': 'unknown', 'zone': 'disc'},
                            'status': 'not_run' if value is None else 'indeterminate',
                            'reason': 'No native IVD crop for this level' if value is None else REASON,
                            'observation': None if value is None else {'label': 'present' if value else 'not_detected',
                                                                       'scale_id': SCALE, 'measurements': []},
                            'image_refs': refs, 'upstream_refs': upstream})
    return {'schema_version': '1.0', 'kind': 'module_result',
            'result_id': 'spinenet-disc-' + manifest_digest(document)[:24],
            'study_id': study['study_id'], 'study_digest': manifest_digest(study), 'synthetic': study['synthetic'],
            'component': 'disc_analysis', 'producer': {'name': 'spinenet-binary-disc', 'version': '1.0', 'method': 'model',
                'model': {'name': 'SpineNet-v2-Herniation', 'version': assets['source_commit'],
                          'weights_sha256': assets['weights']['grading/ckpt1.pt']['sha256']}},
            'assessments': assessments}


def validate_run(study_path, spine_run, directory):
    """Read-only verification; no imports of Torch, NumPy or upstream models."""
    directory, spine_run = Path(directory).resolve(), Path(spine_run).resolve()
    study = load_document(study_path)
    validate_worker_outputs(study, spine_run)
    run = load_document(directory / 'run.json')
    if (not isinstance(run, dict) or run.get('kind') != 'spinenet_binary_disc_run' or run.get('status') != 'completed'
            or run.get('schema_version') != '1.0' or run.get('study_digest') != manifest_digest(study)
            or run.get('inference_performed') is not True or run.get('source_bytes_unchanged') is not True
            or run.get('code_files_sha256') != code_hashes()
            or run.get('input_files_sha256') != input_hashes(study_path, spine_run)
            or run.get('compatibility') != ['numpy.bool=numpy.bool_ in isolated worker']):
        raise ValueError('Disc run inputs, status or code differ')
    assets = verify_assets(run['source'])
    if run.get('assets') != assets:
        raise ValueError('Disc checkpoint/source binding differs')
    doc = load_document(directory / 'disc.json')
    if (not isinstance(doc, dict) or set(doc) != {'schema_version','kind','study_id','study_digest','scale_id','predictions'}
            or doc['schema_version'] != '1.0' or doc['kind'] != 'spinenet_binary_disc'
            or doc['study_id'] != study['study_id'] or doc['study_digest'] != manifest_digest(study)
            or doc['scale_id'] != SCALE or not isinstance(doc['predictions'], list)
            or not all(isinstance(r, dict) for r in doc['predictions'])
            or [r.get('native_level') for r in doc['predictions']] != list(LEVELS)):
        raise ValueError('Invalid native disc document')
    native = load_document(spine_run / 'native.json')['detections']
    geometry = load_document(spine_run / 'spatial.json')['geometry']
    expected_files = {'disc.json', 'result.json'}
    for row in doc['predictions']:
        if (set(row) != {'native_level','native_class','logits','side','size_mm','morphology','zone','source_samples','crop'}
                or any(row[key] is not None for key in ('side','size_mm','morphology','zone'))):
            raise ValueError('Disc grading must not invent side, size or morphology')
        level, value = row['native_level'], row['native_class']
        samples = source_samples(native, geometry, level)
        if value is None:
            if samples is not None or row['logits'] is not None or row['crop'] is not None or row['source_samples'] != []:
                raise ValueError('Missing IVD is not a negative class')
            continue
        logits = row['logits']
        if (type(value) is not int or value not in (0,1) or not isinstance(logits, list) or len(logits) != 2
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in logits)
                or value != (1 if logits[1] > logits[0] else 0) or row['source_samples'] != samples):
            raise ValueError('Disc class, logits or source-frame roster differs')
        crop = row['crop']
        if crop != {'file': 'inputs/' + level + '.f64', 'shape': SHAPE,
                    'image': 'images/' + level + '.png', 'image_plane': 4}:
            raise ValueError('Disc crop identity/plane differs')
        expected_files.update((crop['file'], crop['image']))
    hashes = run.get('artifacts_sha256')
    if not isinstance(hashes, dict) or set(hashes) != expected_files:
        raise ValueError('Disc run must bind every native result and model input')
    files = {**run['input_files_sha256'], str(directory / 'run.json'): file_sha256(directory / 'run.json')}
    for name, digest in hashes.items():
        path = directory / name
        if path.is_symlink() or directory not in path.resolve().parents or file_sha256(path) != digest:
            raise ValueError('Disc artifact SHA mismatch')
        files[str(path.resolve())] = digest
    for row in doc['predictions']:
        if row['crop'] is not None:
            crop = row['crop']
            if (directory / crop['image']).read_bytes() != png_from_input((directory / crop['file']).read_bytes()):
                raise ValueError('Disc illustration differs from actual model input')
    localization = load_document(spine_run / 'localization.json')
    result = load_document(directory / 'result.json')
    if result != build_result(study, localization, doc, assets):
        raise ValueError('C001 disc result differs from native classes')
    validate_bundle(study, [localization, result])
    return doc, result, files
