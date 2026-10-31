"""Portable, read-only caption/summary bundles for fresh evaluation runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile

from malow.video.progress import canonical, digest, atomic_json


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def segment_key(video, times):
    return canonical([Path(video).name, times])


class TextCache:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['format_version'] != 1:
            raise ValueError('Unsupported text-cache bundle version')
        records = self.root / 'records.jsonl'
        if file_sha(records) != self.manifest['records_sha256']:
            raise ValueError('Text-cache index checksum mismatch')
        self.summaries, self.captions, self.verified_videos = {}, {}, set()
        for line in records.open():
            row = json.loads(line)
            for kind in ('caption', 'summary'):
                record = row[kind]
                if record['sha256'] != digest(record['value']):
                    raise ValueError(f'Corrupt bundled {kind}')
            key = row['caption']['key']
            if key != row['summary']['key']:
                raise ValueError('Caption/summary interval mismatch')
            segment = segment_key(key['video'][0], key['frame_times'])
            caption = row['caption']['value']
            if segment in self.captions and self.captions[segment]['caption'] != caption['caption']:
                raise ValueError('Ambiguous captions for the same video interval')
            self.captions[segment] = caption
            identity = (segment, key['previous_memory'], key['revision'])
            summary = row['summary']['value']
            if identity in self.summaries and self.summaries[identity]['memory'] != summary['memory']:
                raise ValueError('Ambiguous summaries for the same memory chain')
            self.summaries[identity] = summary

    def validate_protocol(self, args):
        source = self.manifest['source_config']
        for name in ('rollout_step_seconds', 'base_tail_seconds', 'frame_per_lora',
                     'encoder_video_max_pixels', 'encoder_video_longest_edge', 'video_fps',
                     'interactive_max_segment_frames'):
            if getattr(args, name, None) != source.get(name):
                raise ValueError(f'Text cache uses a different {name}')
        if bool(getattr(args, 'streaming_group_by_video', False)) != bool(source.get('streaming_group_by_video', False)):
            raise ValueError('Text cache uses different video grouping')
        from malow.video.text_summary import experiment_identity
        current = experiment_identity()
        for name in ('prompts_sha256', 'summary_max_tokens', 'caption_max_new_tokens'):
            if source['text_summary_settings'][name] != current[name]:
                raise ValueError(f'Text cache uses different {name}')
        annotation_root = Path(getattr(args, 'annotation_root', None) or args.data_root)
        actual = {str(p.relative_to(annotation_root)): file_sha(p) for p in sorted(annotation_root.rglob('*'))
                  if p.suffix in ('.json', '.parquet')}
        if actual != source['annotation_fingerprints']:
            raise ValueError('Text cache annotations differ from evaluation annotations')

    def get(self, kind, video, times, previous_memory, revision):
        path = Path(video).resolve()
        if str(path) not in self.verified_videos:
            expected = self.manifest['videos'].get(path.name)
            if expected is None or path.stat().st_size != expected['size'] or file_sha(path) != expected['sha256']:
                raise ValueError(f'Text cache video content mismatch: {path}')
            self.verified_videos.add(str(path))
        segment = segment_key(video, times)
        if kind == 'caption':
            return self.captions.get(segment)
        return self.summaries.get((segment, previous_memory, revision))


def init_text_cache(args):
    source = getattr(args, 'text_cache_source', None)
    if not source:
        return
    cache = TextCache(source)
    cache.validate_protocol(args)
    args._external_text_cache = cache
    args._text_cache_identity = file_sha(cache.root / 'manifest.json')
    output = Path(args.output_dir).resolve()
    if output == cache.root or cache.root in output.parents or output in cache.root.parents:
        raise ValueError('Replay output must be separate from its source text-cache bundle')


def export_bundle(source, destination, benchmark):
    source, destination = Path(source), Path(destination)
    config = json.loads((source / 'interactive_run_config.json').read_text())
    destination.mkdir(parents=True, exist_ok=False)
    records, files, videos = [], [], {}
    for caption in sorted((source / 'recovery').glob('*/caption/*.json')):
        # Caption and summary hashes include their kind, so match by full key.
        files.append(caption)
    summaries = {}
    for p in sorted((source / 'recovery').glob('*/summary/*.json')):
        record = json.loads(p.read_text())
        summaries[(p.parent.parent.name, canonical(record['key']))] = (p, record)
    for p in files[:]:
        caption = json.loads(p.read_text())
        pair = summaries.get((p.parent.parent.name, canonical(caption['key'])))
        if pair is None:
            raise ValueError(f'Caption has no completed summary: {p}')
        summary_path, summary = pair
        if any(r['sha256'] != digest(r['value']) for r in (caption, summary)):
            raise ValueError(f'Corrupt source cache: {p}')
        records.append({'caption': caption, 'summary': summary})
        files.append(summary_path)
        video, size, _mtime = caption['key']['video']
        if Path(video).name in videos and videos[Path(video).name]['source_path'] != video:
            raise ValueError('Duplicate video basenames are not portable')
        videos[Path(video).name] = {'source_path': video, 'size': size}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, sha in pool.map(lambda item: (item[0], file_sha(item[1]['source_path'])), videos.items()):
            videos[name]['sha256'] = sha
    index = destination / 'records.jsonl'
    with index.open('w') as stream:
        for row in records: stream.write(canonical(row) + '\n')
    manifest = {'format_version': 1, 'benchmark': benchmark, 'segments': len(records),
                'records_sha256': file_sha(index), 'source_config': config, 'videos': videos}
    atomic_json(destination / 'manifest.json', manifest)
    (destination / 'README.md').write_text(
        f'# MaLoW {benchmark} text-memory cache\n\n'
        f'{len(records)} captions and corresponding compressed summaries, from the completed '
        '30-second fixed-gap Qwen3.5-9B evaluation. Frames are sampled at 1 FPS; '
        'current observations use the last two seconds.\n\n'
        'Extract this archive and pass its directory as --text-cache-source. '
        'records.jsonl is a portable index; recovery/ contains the original individual '
        'caption and summary records. manifest.json records provenance and video content '
        'SHA-256 hashes. No predictions, answer caches, videos, or model weights are included.\n')
    archive = destination.parent / (benchmark + '-text-cache.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in (index, destination / 'manifest.json', destination / 'README.md'):
            z.write(p, p.name)
        for p in files: z.write(p, str(p.relative_to(source)))
    # Round-trip index validation also rejects ambiguous portable keys.
    TextCache(destination)
    return {'archive': str(archive), 'sha256': file_sha(archive), 'segments': len(records), 'bytes': archive.stat().st_size}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--destination', required=True)
    p.add_argument('--benchmark', choices=('ovo', 'streaming'), required=True)
    args = p.parse_args()
    print(json.dumps(export_bundle(args.source, args.destination, args.benchmark)), flush=True)


if __name__ == '__main__': main()
