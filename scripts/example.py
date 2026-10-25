#!/usr/bin/env python3
"""Verify the published example or prepare its portable MRI input for the tool."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mri_root_impact.contracts import load_document, validate_bundle
from mri_fusion.evidence_report import render_html, render_markdown

EXAMPLE = ROOT / 'examples' / 'rsna-3139265320'


def checked_path(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Invalid example path')
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Example paths must stay inside the example directory')
    candidate = root / path
    if candidate.is_symlink() or root.resolve() not in candidate.resolve().parents:
        raise ValueError('Example path escapes its directory')
    return candidate


def checked_example(root=EXAMPLE):
    root = Path(root).resolve()
    inventory = load_document(root / 'files.sha256.json')
    if inventory.get('kind') != 'example_checksums' or inventory.get('study_id') != 'rsna-3139265320':
        raise ValueError('Wrong example inventory')
    files = inventory.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing example checksums')
    for name, expected in files.items():
        path = checked_path(root, name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Example file missing or changed: ' + name)
    study = load_document(root / 'input/study.template.json')
    names = []
    for series in study['series']:
        for image in series['images']:
            relative = 'input/' + image['path']
            path = checked_path(root, relative)
            if files.get(relative) != image['sha256']:
                raise ValueError('Image checksum differs from the manifest: ' + relative)
            names.append(relative)
            image['path'] = str(path)
    if len(names) != 60 or len(set(names)) != 60:
        raise ValueError('The control example requires exactly 60 distinct DICOM files')
    if {name for name in files if name.endswith('.dcm')} != set(names):
        raise ValueError('DICOM inventory and study manifest disagree')
    validate_bundle(study, [])
    fusion = load_document(root / 'output/fusion.json')
    evidence = load_document(root / 'output/report-evidence.json')
    provenance = load_document(root / 'provenance.json')
    if fusion['study_id'] != study['study_id'] or evidence['study_id'] != study['study_id']:
        raise ValueError('Example output belongs to another study')
    if files['output/fusion.json'] != provenance['original_run']['files_sha256']['fusion.json']:
        raise ValueError('Export changed the original fusion result')
    for image in evidence['images'].values():
        image_path = checked_path(root / 'output', image['source_file'])
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != image['sha256']:
            raise ValueError('Output illustration changed')
    for name, renderer in [('report.html', render_html), ('report.md', render_markdown)]:
        if (root / 'output' / name).read_text(encoding='utf-8') != renderer(fusion, evidence):
            raise ValueError('Saved report differs from the production renderer: ' + name)
    return study, inventory


def prepare(output, root=EXAMPLE):
    study, _ = checked_example(root)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Preparation requires a new output directory')
    output = output.resolve()
    if any((p / '.git').exists() for p in (output,) + tuple(output.parents)):
        raise ValueError('Prepared input must be outside Git')
    source = Path(root).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Output overlaps the example source')
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'study.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(study, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return {'ok': True, 'study': str(path), 'dicom_files': 60,
            'source_images_copied': False, 'inference_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify', help='Check example hashes, images and saved report offline')
    prepare_parser = commands.add_parser('prepare', help='Materialize a valid C001 manifest with absolute paths')
    prepare_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.output)
        else:
            study, inventory = checked_example()
            result = {'ok': True, 'study_id': study['study_id'], 'dicom_files': 60,
                      'checked_files': len(inventory['files']), 'report_matches_renderer': True,
                      'inference_performed': False}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'ok': False, 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
