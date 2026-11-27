"""Freeze the reference four-GPU sample schedule, and read it by global position.

JSONL shards are storage chunks, not assignments to GPUs or DataLoader workers.
Every duplicate is an intentional occurrence in the reference training stream.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import hashlib
from importlib.metadata import version
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

from .data import registry, resolve_training_sources, safe_relative, sha256
from .recipes import MIXTURE, SOURCE_COMMIT

FORMAT = "malow-reference-replay"
BATCH = 64
STEPS = 2800
SHARDS = 8
REFERENCE = {
    "source_commit": SOURCE_COMMIT,
    "launcher": "v1_5_1_event_order_r2_model_verified_cross_video_mc_2to1.sbatch",
    "seed": 42, "num_processes": 4, "dataloader_num_workers": 4,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
    "max_steps": STEPS, "mix_strategy": "interleave_over",
    "interleave_probs": MIXTURE, "streaming_repeat": True, "resume": None,
}
BUILD_VERSIONS = {"datasets": "4.0.0", "accelerate": "1.11.0", "torch": "2.6.0"}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def fingerprint(manifest):
    return hashlib.sha256(_json_bytes({k: v for k, v in manifest.items() if k != "fingerprint"})).hexdigest()


def load_manifest(path):
    path = Path(path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("version") != 1:
        raise ValueError("Unsupported replay manifest format.")
    if manifest.get("fingerprint") != fingerprint(manifest):
        raise ValueError("Replay manifest fingerprint mismatch.")
    count = manifest["occurrences"]
    if (manifest["global_batch_size"] != BATCH or count != manifest["steps"] * BATCH
            or count < BATCH * SHARDS or count % (BATCH * SHARDS)
            or len(manifest["shards"]) != SHARDS):
        raise ValueError("Replay requires eight equal shards containing complete 64-sample steps.")
    if len(manifest["step_sha256"]) != manifest["steps"]:
        raise ValueError("Replay step digest count mismatch.")
    names = []
    for i, shard in enumerate(manifest["shards"]):
        if shard["start"] != i * (count // SHARDS) or shard["count"] != count // SHARDS:
            raise ValueError("Replay shard positions overlap or leave gaps.")
        for key in ("file", "index"):
            name = shard[key]
            safe_relative(name)
            if not (path.parent / name).resolve().is_relative_to(path.parent):
                raise ValueError("Replay file escapes its directory.")
            names.append(name)
    if len(set(names)) != SHARDS * 2:
        raise ValueError("Replay file names must be distinct.")
    for source in manifest["sources"]:
        safe_relative(source["video_root"])
        for shard in source["shards"]:
            safe_relative(shard["file"])
    derivation = manifest.get("derivation")
    if derivation is not None:
        if (derivation.get("kind") != "ovo-filtered-sft" or derivation.get("schema") != "malow-replay-sft-v1"
                or derivation.get("supported_loss") != "sft_expected_or_generated"
                or derivation.get("requires_base_with_last_2seconds") is not True):
            raise ValueError("Unsupported derived replay schema or training contract.")
        for name in derivation.get("audit_sha256", {}):
            safe_relative(name)
            if not (path.parent / name).resolve().is_relative_to(path.parent):
                raise ValueError("Replay audit file escapes its directory.")
    return manifest


def validate_training_contract(manifest, loss_type, base_with_last_2seconds):
    """Cleaned answers and omitted teacher frame counts support the current SFT recipes only."""
    if manifest.get("derivation") and (loss_type != "sft_expected_or_generated" or not base_with_last_2seconds):
        raise ValueError("Cleaned replay requires sft_expected_or_generated and base_with_last_2seconds.")


def validate_layout(world_size, batch_size, accumulation, max_steps, manifest):
    if world_size not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("Replay GPU count must be one of 1, 2, 4, 8, 16, 32, 64.")
    if batch_size != 1 or accumulation != BATCH // world_size:
        raise ValueError("Replay requires batch size 1 and gradient accumulation 64 / GPU count.")
    if not 0 < max_steps <= manifest["steps"]:
        raise ValueError(f"Replay max_steps must be between 1 and {manifest['steps']}.")


def _identity_bytes(identity):
    return struct.pack("<III", *identity)


def _identity_rows(shards, stream):
    """One generator shard per original file, including its exact retained rows."""
    for shard in shards:
        for row in shard["accepted"]:
            yield {"source": (stream, shard["number"], row)}


def _first(rows):
    return rows[0]["source"]


def reference_schedule(source_shards, steps=STEPS):
    """Run pinned HF/Accelerate/PyTorch sharding and worker scheduling on row IDs.

    Filtering has already been evaluated on every original record. The omitted
    canonicalization and preprocessing maps are one-to-one and do not use the
    interleaver's RNG. No hand-written emulation of shard or worker ordering.
    """
    import numpy as np
    from accelerate.data_loader import prepare_data_loader
    from datasets import IterableDataset, interleave_datasets
    from torch.utils.data import DataLoader

    for package, expected in BUILD_VERSIONS.items():
        if version(package).split("+")[0] != expected:
            raise RuntimeError(f"Replay build requires {package}=={expected}; found {version(package)}.")
    sources = [IterableDataset.from_generator(_identity_rows, gen_kwargs={"shards": shards, "stream": i})
               for i, shards in enumerate(source_shards)]
    full = interleave_datasets(sources, probabilities=list(map(float, MIXTURE.split(","))),
                              seed=42, stopping_strategy="all_exhausted").repeat(None)
    if full.n_shards != SHARDS:
        raise ValueError("Reference loader must expose eight source shards.")
    ranks = []
    for rank in range(4):
        loader = prepare_data_loader(
            DataLoader(copy.deepcopy(full), batch_size=1, num_workers=4, collate_fn=_first),
            num_processes=4, process_index=rank, dispatch_batches=False, split_batches=False)
        iterator = iter(loader)
        try:
            rows = list(islice(iterator, steps * 16))
        finally:
            # Closing the generator releases nonpersistent worker processes.
            iterator.close()
        if len(rows) != steps * 16:
            raise ValueError(f"Reference rank {rank} exhausted before the requested schedule.")
        ranks.append(rows)
    # rank, step, microstep, identity -> step, microstep, rank, identity
    return np.asarray(ranks, dtype=np.int64).reshape(4, steps, 16, 3).transpose(1, 2, 0, 3).reshape(-1, 3)


def _scan_sources(annotations_root, videos_root):
    from ._vendor.llamafactory.train.c2l.data import (
        _get_profile_files, _infer_source_video_root, _row_has_existing_videos,
    )

    spec = registry()
    profiles, roots = resolve_training_sources(annotations_root, videos_root)
    sources, scheduling, offsets = [], [], {}
    for stream, (description, profile, root) in enumerate(zip(spec["streams"], profiles, roots, strict=True)):
        paths = [Path(path) for path in _get_profile_files(profile)]
        if len(paths) != SHARDS or any(path.suffix != ".jsonl" for path in paths):
            raise ValueError(f"{profile}: expected exactly eight JSONL source shards.")
        source_root = _infer_source_video_root(profile)
        source = {"id": description["id"], "profile_dir": description["profile_dir"],
                  "video_root": spec["video_roots"][description["video_source"]],
                  "source_video_root": source_root, "answer_instruction": "", "shards": []}
        schedule = []
        for shard, path in enumerate(paths):
            positions, accepted = [], []
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row_number = len(positions)
                    positions.append(offset)
                    if _row_has_existing_videos(row, str(videos_root), root, source_root):
                        accepted.append(row_number)
            if not accepted:
                raise ValueError(f"Empty source shard after video filtering: {path}")
            source["shards"].append({
                "file": path.relative_to(annotations_root).as_posix(), "sha256": digest.hexdigest(),
                "rows": len(positions), "retained": len(accepted), "filtered": len(positions) - len(accepted),
                "retained_rows_sha256": hashlib.sha256(_json_bytes(accepted)).hexdigest(),
            })
            schedule.append({"number": shard, "accepted": accepted})
            offsets[stream, shard] = positions
        sources.append(source)
        scheduling.append(schedule)
        print(f"Scanned {source['id']}: {sum(s['retained'] for s in source['shards'])} retained rows", flush=True)
    return sources, scheduling, offsets


def write_replay(directory, identities, sources, read_record):
    """Write an indexed schedule into a new staging directory; never deduplicate."""
    import numpy as np

    count = len(identities)
    if count % (SHARDS * BATCH):
        raise ValueError("Schedule must divide into eight whole-step shards.")
    per_shard = count // SHARDS
    manifest = {
        "format": FORMAT, "version": 1, "reference": REFERENCE,
        "versions": {p: version(p) for p in (*BUILD_VERSIONS, "transformers", "numpy")},
        "global_batch_size": BATCH, "steps": count // BATCH, "occurrences": count,
        "position_order": "step * 64 + original_microstep * 4 + original_rank",
        "sources": sources, "shards": [], "step_sha256": [],
        "recipe_provenance_sha256": sha256(Path(__file__).with_name("recipe_provenance.json")),
    }
    schedule_hash, step_hash = hashlib.sha256(), hashlib.sha256()
    counts, unique = Counter(), set()
    for shard_number in range(SHARDS):
        name = f"replay-{shard_number:05d}-of-00008.jsonl"
        index_name = name + ".offsets.npy"
        offsets = []
        path = directory / name
        with path.open("wb") as handle:
            for position in range(shard_number * per_shard, (shard_number + 1) * per_shard):
                identity = tuple(map(int, identities[position]))
                raw = read_record(identity).strip()
                # Keep the original JSON record, including its nested teacher payload.
                prefix = _json_bytes({"position": position, "source": identity})[:-1]
                offsets.append(handle.tell())
                handle.write(prefix + b',"record":' + raw + b"}\n")
                encoded = _identity_bytes(identity)
                schedule_hash.update(encoded)
                step_hash.update(encoded)
                counts[str(identity[0])] += 1
                unique.add(identity)
                if (position + 1) % BATCH == 0:
                    manifest["step_sha256"].append(step_hash.hexdigest())
                    step_hash = hashlib.sha256()
            offsets.append(handle.tell())
        np.save(directory / index_name, np.asarray(offsets, dtype="<u8"), allow_pickle=False)
        manifest["shards"].append({"file": name, "index": index_name, "start": shard_number * per_shard,
                                   "count": per_shard, "sha256": sha256(path),
                                   "index_sha256": sha256(directory / index_name)})
    manifest.update(schedule_sha256=schedule_hash.hexdigest(), counts_by_stream=dict(counts),
                    unique_source_rows=len(unique))
    manifest["fingerprint"] = fingerprint(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build(annotations_root, videos_root, output_dir):
    annotations_root = Path(annotations_root).expanduser().resolve()
    videos_root = Path(videos_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Replay output already exists: {output_dir}")
    sources, scheduling, offsets = _scan_sources(annotations_root, videos_root)
    print("Replaying four reference ranks with four configured workers per rank...", flush=True)
    identities = reference_schedule(scheduling)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        with ExitStack() as stack:
            handles = {(i, j): stack.enter_context((annotations_root / shard["file"]).open("rb"))
                       for i, source in enumerate(sources) for j, shard in enumerate(source["shards"])}

            def read_record(identity):
                stream, shard, row = identity
                handle = handles[stream, shard]
                handle.seek(offsets[stream, shard][row])
                return handle.readline()

            write_replay(staging, identities, sources, read_record)
        # Catch source changes during build rather than blessing a mixed snapshot.
        for source in sources:
            for shard in source["shards"]:
                if sha256(annotations_root / shard["file"]) != shard["sha256"]:
                    raise ValueError(f"Source changed during replay build: {shard['file']}")
        report = validate(staging / "manifest.json", videos_root=videos_root)
        # mkdtemp starts owner-only; match the parent dataset directory on publish.
        staging.chmod(output_dir.parent.stat().st_mode & 0o777)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), **report}, indent=2), flush=True)
    return output_dir / "manifest.json"


class ReplayDataset:
    """Map-style, random-access records. Handles are private to each worker PID."""

    def __init__(self, manifest_path, transform=None):
        import numpy as np

        self.path = Path(manifest_path).expanduser().resolve()
        self.manifest = load_manifest(self.path)
        self.transform = transform
        self._indexes = []
        self._handles = {}
        self._pid = os.getpid()
        for shard in self.manifest["shards"]:
            index = np.load(self.path.parent / shard["index"], allow_pickle=False)
            if index.dtype != np.dtype("<u8") or index.shape != (shard["count"] + 1,):
                raise ValueError(f"Invalid replay offset index: {shard['index']}")
            if index[0] != 0 or not (index[1:] > index[:-1]).all():
                raise ValueError(f"Replay offsets must increase: {shard['index']}")
            if int(index[-1]) != (self.path.parent / shard["file"]).stat().st_size:
                raise ValueError(f"Replay offset/file size mismatch: {shard['file']}")
            self._indexes.append(index)

    def __len__(self):
        return self.manifest["occurrences"]

    def __getstate__(self):
        return {**self.__dict__, "_handles": {}, "_pid": None}

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        if hasattr(self, "_handles"):
            self.close()

    def __getitem__(self, position):
        if not 0 <= position < len(self):
            raise IndexError(position)
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        shard_number, row = divmod(position, len(self) // SHARDS)
        shard = self.manifest["shards"][shard_number]
        if shard_number not in self._handles:
            self._handles[shard_number] = (self.path.parent / shard["file"]).open("rb")
        handle = self._handles[shard_number]
        start, end = map(int, self._indexes[shard_number][row:row + 2])
        handle.seek(start)
        envelope = json.loads(handle.read(end - start))
        if envelope["position"] != position:
            raise ValueError(f"Replay index points to the wrong occurrence: {position}")
        return self.transform(envelope) if self.transform else envelope


def validate(manifest_path, videos_root=None):
    """Check every file, offset, identity, step digest and optionally video existence."""
    dataset = ReplayDataset(manifest_path)
    manifest = dataset.manifest
    try:
        for shard in manifest["shards"]:
            for filename, digest in (("file", "sha256"), ("index", "index_sha256")):
                if sha256(dataset.path.parent / shard[filename]) != shard[digest]:
                    raise ValueError(f"Replay checksum mismatch: {shard[filename]}")
        for name, digest in manifest.get("derivation", {}).get("audit_sha256", {}).items():
            if sha256(dataset.path.parent / name) != digest:
                raise ValueError(f"Replay audit checksum mismatch: {name}")
        if videos_root is not None:
            from ._vendor.llamafactory.train.c2l.data import _row_has_existing_videos
            videos_root = Path(videos_root).expanduser().resolve()
        schedule_hash, step_hash = hashlib.sha256(), hashlib.sha256()
        counts, unique, checked = Counter(), set(), set()
        for position in range(len(dataset)):
            envelope = dataset[position]
            identity = tuple(envelope["source"])
            if len(identity) != 3 or any(type(i) is not int or i < 0 for i in identity):
                raise ValueError(f"Invalid source identity at position {position}")
            stream, shard, row = identity
            try:
                source = manifest["sources"][stream]
                if row >= source["shards"][shard]["rows"]:
                    raise IndexError(row)
            except IndexError as exc:
                raise ValueError(f"Out-of-range source identity: {identity}") from exc
            if not isinstance(envelope["record"], dict):
                raise ValueError(f"Invalid record at position {position}")
            encoded = _identity_bytes(identity)
            schedule_hash.update(encoded)
            step_hash.update(encoded)
            counts[str(stream)] += 1
            unique.add(identity)
            if videos_root is not None and identity not in checked:
                if not _row_has_existing_videos(envelope["record"], str(videos_root),
                                                str(videos_root / source["video_root"]),
                                                source["source_video_root"]):
                    raise FileNotFoundError(f"Missing replay video for source row {identity}")
                checked.add(identity)
            if (position + 1) % BATCH == 0:
                if step_hash.hexdigest() != manifest["step_sha256"][position // BATCH]:
                    raise ValueError(f"Replay step digest mismatch at step {position // BATCH}")
                step_hash = hashlib.sha256()
        if (schedule_hash.hexdigest() != manifest["schedule_sha256"]
                or dict(counts) != manifest["counts_by_stream"]
                or len(unique) != manifest["unique_source_rows"]):
            raise ValueError("Replay schedule statistics/digest mismatch.")
        return {"replay_fingerprint": manifest["fingerprint"], "occurrences": len(dataset),
                "unique_source_rows": len(unique), "steps": manifest["steps"],
                "annotation_sha256": {s["file"]: s["sha256"] for s in manifest["shards"]}}
    finally:
        dataset.close()


class ProfileTransform:
    """Use the same canonicalization and preprocessing as the streaming recipe."""

    def __init__(self, sources, template, tokenizer, processor, data_args, finetuning_args):
        self.sources = sources
        self.template, self.tokenizer, self.processor = template, tokenizer, processor
        self.data_args, self.finetuning_args = data_args, finetuning_args

    def __call__(self, envelope):
        from ._vendor.llamafactory.train.c2l.data import (
            _row_has_existing_videos, preprocess_profile_record,
        )
        source = self.sources[envelope["source"][0]]
        root = str(Path(self.data_args.media_dir) / source["video_root"])
        row = envelope["record"]
        if not _row_has_existing_videos(row, self.data_args.media_dir, root, source["source_video_root"]):
            raise FileNotFoundError(f"Missing replay video at position {envelope['position']}; cannot skip.")
        canonical = prepare_profile_record(row, self.finetuning_args.c2l_loss_type)
        return preprocess_profile_record(
            canonical, self.template, self.tokenizer, self.processor, self.data_args, self.finetuning_args,
            answer_instruction=source["answer_instruction"] or None, video_root=root,
            source_video_root=source["source_video_root"], video_merge_candidates=None, video_merge_seed=42)


def prepare_profile_record(row, loss_type):
    """Match a schema-unknown JSON IterableDataset.map replacement exactly.

    HF map retains raw columns outside remove_columns, including top-level clip
    metadata. Calling just the canonicalizer would silently discard those fields.
    """
    from ._vendor.llamafactory.train.c2l.data import (
        _canonicalize_streaming_profile_record, _sft_map_remove_columns,
    )

    canonical = _canonicalize_streaming_profile_record(row, loss_type)
    if loss_type == "kl":
        return canonical
    removed = set(_sft_map_remove_columns(None))
    return {**{k: v for k, v in row.items() if k not in removed}, **canonical}


def training_dataset(template, tokenizer, processor, data_args, finetuning_args, training_args):
    manifest = load_manifest(data_args.replay_manifest)
    validate_training_contract(manifest, finetuning_args.c2l_loss_type, finetuning_args.base_with_last_2seconds)
    if data_args.replay_fingerprint != manifest["fingerprint"]:
        raise ValueError("Replay manifest changed since launch.")
    validate_layout(training_args.world_size, training_args.per_device_train_batch_size,
                    training_args.gradient_accumulation_steps, training_args.max_steps, manifest)
    accelerator = training_args.accelerator_config
    if (data_args.streaming or not finetuning_args.disable_shuffling
            or finetuning_args.video_merge_extend or training_args.ignore_data_skip
            or accelerator.dispatch_batches or accelerator.split_batches
            or accelerator.use_seedable_sampler or getattr(accelerator, "use_stateful_dataloader", False)
            or training_args.auto_find_batch_size or training_args.dataloader_drop_last):
        raise ValueError("Replay requires sequential map-style loading, standard batch sharding and Trainer resume skipping.")
    return ReplayDataset(data_args.replay_manifest,
                         ProfileTransform(manifest["sources"], template, tokenizer, processor,
                                          data_args, finetuning_args))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build", help="Freeze the 2800-step reference schedule into eight shards.")
    builder.add_argument("--annotations-root", required=True, type=Path)
    builder.add_argument("--videos-root", required=True, type=Path)
    builder.add_argument("--output-dir", required=True, type=Path)
    validator = commands.add_parser("validate", help="Verify frozen annotations and indexes.")
    validator.add_argument("--manifest", required=True, type=Path)
    validator.add_argument("--videos-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.annotations_root, args.videos_root, args.output_dir)
    else:
        print(json.dumps(validate(args.manifest, args.videos_root), indent=2))


if __name__ == "__main__":
    main()
