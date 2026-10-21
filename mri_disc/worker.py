"""Isolated CPU SpineNet grading worker; invoked only for explicit disc analysis."""

import argparse
from pathlib import Path
import sys
import time

from mri_root_impact.contracts import load_document, manifest_digest
from mri_root_impact.sagittal import load_sagittal_series
from mri_root_impact.spinenet_assets import file_sha256, verify_assets
from mri_root_impact.spinenet_runner import external_run_directory, validate_worker_outputs
from mri_root_impact.spinenet_worker import runtime_versions
from .contract import LEVELS, SCALE, SHAPE, build_result, code_hashes, input_hashes, png_from_input, source_samples, write


def run(study_path, spine_run, source, output, allow_missing_modality=False, threads=2):
    study_path, spine_run, source = (Path(p).resolve() for p in (study_path, spine_run, source))
    study = load_document(study_path)
    if type(threads) is not int or not 1 <= threads <= 8 or type(allow_missing_modality) is not bool:
        raise ValueError('Invalid disc worker parameters')
    output = external_run_directory(study_path, study, output, source)
    external_run_directory(study_path, study, output, spine_run)
    runtime = runtime_versions()
    assets = verify_assets(source)
    validate_worker_outputs(study, spine_run)
    bindings = input_hashes(study_path, spine_run)
    output.mkdir(parents=True)
    started = time.monotonic()
    try:
        volume, geometry = load_sagittal_series(study, allow_missing_modality=allow_missing_modality)
        if geometry != load_document(spine_run / 'spatial.json')['geometry']:
            raise ValueError('Disc input geometry differs from localization')
        native = load_document(spine_run / 'native.json')['detections']
        import numpy as np
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0); np.random.seed(0)
        # Upstream IVD masks still use the removed NumPy alias. Worker-only shim.
        if 'bool' in np.__dict__:
            raise ValueError('Unexpected NumPy runtime for pinned disc compatibility')
        np.bool = np.bool_
        sys.dont_write_bytecode = True
        sys.pycache_prefix = str(output / '.source-pycache')
        sys.path.insert(0, str(source))
        from spinenet import SpineNet
        from spinenet.models.grading import GradingModel
        network = SpineNet.__new__(SpineNet)
        network.device = 'cpu'
        network.grading_model = GradingModel()
        network.grading_model.load_weights(str(source / 'spinenet/weights/grading'), verbose=False)
        network.grading_model.eval()
        ivds = network.get_ivds_from_vert_dicts(native, volume)
        levels = [item['level_name'] for item in ivds]
        if len(levels) != len(set(levels)):
            raise ValueError('Duplicate IVD native levels')
        rows = []
        (output / 'inputs').mkdir(); (output / 'images').mkdir()
        for level in LEVELS:
            item = next((item for item in ivds if item['level_name'] == level), None)
            row = {'native_level': level, 'native_class': None, 'logits': None, 'side': None,
                   'size_mm': None, 'morphology': None, 'zone': None, 'source_samples': [], 'crop': None}
            if item is not None:
                crop = item['volume']
                if list(crop.shape) != SHAPE or not np.isfinite(crop).all():
                    raise ValueError('Invalid IVD model input')
                raw = np.asarray(crop, dtype='<f8').tobytes()
                image = png_from_input(raw)
                with torch.inference_mode():
                    logits = network.grading_model(torch.tensor(crop)[None,None].float())[10].squeeze().tolist()
                row.update(native_class=1 if logits[1] > logits[0] else 0, logits=logits,
                           source_samples=source_samples(native, geometry, level),
                           crop={'file': 'inputs/' + level + '.f64', 'shape': SHAPE,
                                 'image': 'images/' + level + '.png', 'image_plane': 4})
                (output / row['crop']['file']).write_bytes(raw)
                (output / row['crop']['image']).write_bytes(image)
            rows.append(row)
        if not any(row['native_class'] is not None for row in rows):
            raise ValueError('No standard lumbar IVD model input was available')
        document = {'schema_version':'1.0','kind':'spinenet_binary_disc','study_id':study['study_id'],
                    'study_digest':manifest_digest(study),'scale_id':SCALE,'predictions':rows}
        result = build_result(study, load_document(spine_run / 'localization.json'), document, assets)
        write(output / 'disc.json', document); write(output / 'result.json', result)
        if bindings != input_hashes(study_path, spine_run) or assets != verify_assets(source):
            raise ValueError('Disc sources changed during inference')
        artifacts = ['disc.json','result.json'] + [r['crop'][key] for r in rows if r['crop'] for key in ('file','image')]
        write(output / 'run.json', {'schema_version':'1.0','kind':'spinenet_binary_disc_run','status':'completed',
              'study_digest':manifest_digest(study),'inference_performed':True,'source_bytes_unchanged':True,
              'source':str(source),'assets':assets,'runtime':runtime,'code_files_sha256':code_hashes(),
              'input_files_sha256':bindings,'artifacts_sha256':{n:file_sha256(output / n) for n in artifacts},
              'parameters':{'allow_missing_modality':allow_missing_modality,'threads':threads},
              'compatibility':['numpy.bool=numpy.bool_ in isolated worker'],
              'duration_seconds':time.monotonic()-started})
    except Exception as exc:
        write(output / 'run.json', {'kind':'spinenet_binary_disc_run','status':'error','error':str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study','spine-run','source','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--allow-missing-modality', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args.study,args.spine_run,args.source,args.output,args.allow_missing_modality,args.threads)


if __name__ == '__main__':
    main()
