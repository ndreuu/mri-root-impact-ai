#!/usr/bin/env python3
"""Fetch one native RSNA 2024 study from a pinned public HF ZIP by HTTP ranges.

No credentials, terms acceptance, or full archive download. The mirror has no
license card; original RSNA challenge terms still apply. Whole archive SHA256
is recorded as a source identifier, never claimed to have been verified.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import socket
import struct
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path, PurePosixPath

REPO = "darkxtranger/RSNA"
REVISION = "e881937c3222a5fc1c09d573adac3b9a1ff50d35"
ARCHIVE = "rsna-2024-lumbar-spine-degenerative-classification.zip"
URL = "https://huggingface.co/datasets/{}/resolve/{}/{}?download=true".format(REPO, REVISION, ARCHIVE)
ARCHIVE_SIZE = 30303207710
ARCHIVE_SHA256 = "b1507edb2cf0693938e48f173ae61a68ff04cd4128952707ee61adfc32b2d037"
SOURCE = "https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification"
DESCRIPTIONS = ("Sagittal T2/STIR", "Sagittal T1", "Axial T2")
CSVS = ("train_series_descriptions.csv", "train_label_coordinates.csv", "train.csv")
MAX_INDEX = 64 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
MAX_MEMBER = 16 * 1024 * 1024


def report(phase, **details):
    print(json.dumps(dict(phase=phase, **details)), file=sys.stderr, flush=True)


def safe_path(value):
    parts = value.split("/")
    if "\\" in value or any(p in ("", ".", "..") for p in parts) or value.startswith("/"):
        raise ValueError("Unsafe archive or provenance path")
    return PurePosixPath(*parts)


def local_path(root, name):
    path = root.joinpath(*safe_path(name).parts)
    if root.resolve() not in path.resolve().parents:
        raise ValueError("Destination escapes output directory")
    return path


def outside_git(root):
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
        raise ValueError("Store downloaded study outside every Git repository")


class RangeReader:
    def __init__(self, opener=None):
        self.opener = opener or urllib.request.urlopen
        self.total = 0
        self.url = URL

    def read(self, start, length):
        if not 0 <= start < ARCHIVE_SIZE or not 0 < length <= MAX_INDEX:
            raise ValueError("Invalid or oversized archive byte range")
        end = start + length - 1
        if end >= ARCHIVE_SIZE:
            raise ValueError("Archive range download budget exceeded")
        for attempt in (1, 2):
            if self.total + length > MAX_TOTAL:
                raise ValueError("Archive range download budget exceeded")
            # Reserve attempted bytes too: a timed-out read may have received a
            # partial body, so retries must not silently exceed the total cap.
            self.total += length
            request = urllib.request.Request(self.url, headers={
                "Range": "bytes={}-{}".format(start, end),
                "Accept-Encoding": "identity", "User-Agent": "mri-root-impact-control/1",
            })
            try:
                with self.opener(request, timeout=30) as response:
                    expected = "bytes {}-{}/{}".format(start, end, ARCHIVE_SIZE)
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server must return the exact requested HTTP 206 range")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ValueError("Unexpected HTTP content encoding")
                    self.url = getattr(response, "url", self.url)
                    raw = response.read(length + 1)
                if len(raw) != length:
                    raise ValueError("Archive range has an unexpected length")
                return raw
            except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
                timed_out = isinstance(error, (TimeoutError, socket.timeout)) or isinstance(getattr(error, "reason", None), (TimeoutError, socket.timeout))
                if not timed_out or attempt == 2:
                    raise
                report("range_retry", start=start, bytes=length, attempt=attempt + 1)


def archive_index(reader):
    tail_start = ARCHIVE_SIZE - 65557
    report("archive_tail", bytes=65557)
    tail = reader.read(tail_start, 65557)
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0 or pos + 22 > len(tail):
        raise ValueError("ZIP end directory missing")
    end = struct.unpack_from("<4s4H2LH", tail, pos)
    disk, directory_disk, on_disk, count, size, offset, comment = end[1:]
    if pos + 22 + comment != len(tail) or disk or directory_disk:
        raise ValueError("Unexpected ZIP directory layout")
    if count == 65535 or size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        locator = pos - 20
        if locator < 0 or tail[locator:locator + 4] != b"PK\x06\x07":
            raise ValueError("ZIP64 locator missing")
        _, locator_disk, position, disks = struct.unpack_from("<4sLQL", tail, locator)
        if locator_disk or disks != 1:
            raise ValueError("Multi-disk ZIP unsupported")
        block = reader.read(position, 56)
        values = struct.unpack("<4sQ2H2L4Q", block)
        if values[0] != b"PK\x06\x06" or values[4] or values[5] or values[6] != values[7]:
            raise ValueError("Invalid ZIP64 directory")
        count, size, offset = values[7:]
    elif on_disk != count:
        raise ValueError("Split ZIP directory unsupported")
    if not 0 < count <= 1000000 or not 0 < size <= MAX_INDEX:
        raise ValueError("ZIP directory exceeds bounded index limits")
    report("central_directory", bytes=size, entries=count)
    directory = reader.read(offset, size)
    entries, cursor = {}, 0
    while cursor < len(directory):
        if cursor + 46 > len(directory):
            raise ValueError("Truncated ZIP directory")
        item = struct.unpack_from("<4s6H3L5H2L", directory, cursor)
        if item[0] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP directory signature")
        flags, method, crc, compressed, raw = item[3], item[4], item[7], item[8], item[9]
        name_len, extra_len, comment_len, disk_start, offset = item[10:14] + (item[16],)
        stop = cursor + 46 + name_len + extra_len + comment_len
        if stop > len(directory) or disk_start:
            raise ValueError("Invalid ZIP member layout")
        name = directory[cursor + 46:cursor + 46 + name_len].decode("utf-8" if flags & 2048 else "cp437")
        extra = directory[cursor + 46 + name_len:cursor + 46 + name_len + extra_len]
        sentinels = [raw == 0xFFFFFFFF, compressed == 0xFFFFFFFF, offset == 0xFFFFFFFF]
        if any(sentinels):
            extra_pos, found = 0, False
            while extra_pos + 4 <= len(extra):
                kind, length = struct.unpack_from("<HH", extra, extra_pos)
                field = extra[extra_pos + 4:extra_pos + 4 + length]
                extra_pos += length + 4
                if kind == 1:
                    numbers = [raw, compressed, offset]
                    field_pos = 0
                    for index, needed in enumerate(sentinels):
                        if needed:
                            if field_pos + 8 > len(field):
                                raise ValueError("Truncated ZIP64 member")
                            numbers[index] = struct.unpack_from("<Q", field, field_pos)[0]
                            field_pos += 8
                    raw, compressed, offset = numbers
                    found = True
                    break
            if not found:
                raise ValueError("ZIP64 member extension missing")
        if name.endswith("/"):
            safe_path(name[:-1])
        else:
            safe_path(name)
        if name in entries or flags & 1:
            raise ValueError("Duplicate or encrypted ZIP member")
        entries[name] = {"name": name, "offset": offset, "size": raw,
                         "compressed": compressed, "crc": crc, "method": method}
        cursor = stop
    if len(entries) != count:
        raise ValueError("ZIP directory member count mismatch")
    return entries


def read_member(reader, item):
    if not 0 < item["size"] <= MAX_MEMBER or not 0 < item["compressed"] <= MAX_MEMBER:
        raise ValueError("ZIP member exceeds sample limits")
    header = struct.unpack("<4s5H3L2H", reader.read(item["offset"], 30))
    if header[0] != b"PK\x03\x04" or header[2] & 1 or header[3] != item["method"]:
        raise ValueError("Invalid local ZIP member header")
    body = reader.read(item["offset"] + 30, header[-2] + header[-1] + item["compressed"])
    name = body[:header[-2]].decode("utf-8" if header[2] & 2048 else "cp437")
    if name != item["name"]:
        raise ValueError("ZIP local filename mismatch")
    compressed = body[header[-2] + header[-1]:]
    if item["method"] == zipfile.ZIP_DEFLATED:
        decompressor = zlib.decompressobj(-15)
        raw = decompressor.decompress(compressed, item["size"] + 1)
        if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
            raise ValueError("Invalid or oversized deflated member")
    elif item["method"] == zipfile.ZIP_STORED:
        raw = compressed
    else:
        raise ValueError("Unsupported ZIP compression")
    if len(raw) != item["size"] or zlib.crc32(raw) & 0xFFFFFFFF != item["crc"]:
        raise ValueError("ZIP member size/CRC mismatch")
    return raw


class BufferedRange:
    def __init__(self, start, raw):
        self.start, self.raw = start, raw

    def read(self, start, length):
        offset = start - self.start
        if offset < 0 or length <= 0 or offset + length > len(self.raw):
            raise ValueError("ZIP member exceeds fetched block")
        return self.raw[offset:offset + length]


def grouped_members(reader, items):
    """One HTTP request per nearby group; local header/CRC checks stay intact."""
    groups = []
    for item in sorted(items, key=lambda value: value["offset"]):
        # Local filename and extra fields each have at most 65535 bytes.
        end = min(ARCHIVE_SIZE, item["offset"] + 30 + 2 * 65535 + item["compressed"])
        if end - item["offset"] > MAX_INDEX:
            raise ValueError("DICOM range group exceeds bounded block limit")
        if groups and item["offset"] - groups[-1][1] <= 1024 * 1024 and end - groups[-1][0] <= MAX_INDEX:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append(item)
        else:
            groups.append([item["offset"], end, [item]])
    for start, end, group in groups:
        report("dicom_block", bytes=end - start, members=len(group))
        buffered = BufferedRange(start, reader.read(start, end - start))
        for item in group:
            yield item, read_member(buffered, item)


def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def choose_study(entries, series, coordinates, study_id=None):
    grouped, frames, coverage = defaultdict(list), defaultdict(list), defaultdict(set)
    for name, item in entries.items():
        parts = name.split("/")
        if len(parts) == 4 and parts[0] == "train_images" and name.endswith(".dcm"):
            frames[(parts[1], parts[2])].append(item)
    for row in coordinates:
        coverage[(row["study_id"], row["series_id"])].add((row["condition"], row["level"]))
    for row in series:
        if row["series_description"] in DESCRIPTIONS and (not study_id or row["study_id"] == study_id):
            if row["study_id"].isdigit() and row["series_id"].isdigit():
                grouped[row["study_id"]].append(row)
    candidates = []
    for study, rows in grouped.items():
        chosen = []
        for description in DESCRIPTIONS:
            options = [r for r in rows if r["series_description"] == description and frames[(study, r["series_id"])]]
            if not options:
                break
            chosen.append(min(options, key=lambda r: (-len(coverage[(study, r["series_id"])]),
                              sum(i["compressed"] for i in frames[(study, r["series_id"])]), r["series_id"])))
        if len(chosen) != 3:
            continue
        selected = [item for row in chosen for item in frames[(study, row["series_id"])]]
        size = sum(i["size"] for i in selected)
        if size > MAX_TOTAL // 2 or any(i["size"] > MAX_MEMBER for i in selected):
            continue
        score = len(set().union(*(coverage[(study, r["series_id"])] for r in chosen)))
        candidates.append((-score, size, study, chosen, sorted(selected, key=lambda i: i["name"])))
    if not candidates:
        raise ValueError("No bounded study with all three required native series found")
    _, _, study, chosen, selected = min(candidates, key=lambda item: item[:3])
    ids = {r["series_id"] for r in chosen}
    labels = [r for r in coordinates if r["study_id"] == study and r["series_id"] in ids]
    return study, chosen, labels, selected


def atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    try:
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def subset_csv(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def verify_existing(root):
    try:
        provenance = json.loads((root / "provenance.json").read_text())
        if (provenance["repo_id"] != REPO or provenance["revision"] != REVISION
                or provenance["archive_sha256"] != ARCHIVE_SHA256
                or provenance["whole_archive_sha256_verified"] is not False
                or not provenance["study_id"].isdigit()):
            return None
        seen, counts = set(), defaultdict(int)
        if not provenance["files"] or {i["relative_path"] for i in provenance["csv_files"]} != set(CSVS):
            return None
        for item in provenance["files"] + provenance["csv_files"]:
            name = item["relative_path"]
            parts = safe_path(name).parts
            if name in seen or item["zip_crc_verified"] is not True:
                return None
            seen.add(name)
            if name not in CSVS:
                if len(parts) != 4 or parts[0] != "train_images" or parts[1] != provenance["study_id"] or not parts[2].isdigit() or not parts[3].endswith(".dcm"):
                    return None
                counts[parts[2]] += 1
            raw = local_path(root, name).read_bytes()
            if len(raw) != item["size_bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                return None
        rows = csv_rows((root / CSVS[0]).read_bytes())
        if len(rows) != 3 or {r["series_description"] for r in rows} != set(DESCRIPTIONS):
            return None
        if {r["series_id"] for r in rows} != set(counts) or any(r["study_id"] != provenance["study_id"] for r in rows) or not set(CSVS) <= seen:
            return None
        if provenance["series"] != [{"series_id": r["series_id"], "series_description": r["series_description"]} for r in rows]:
            return None
        return provenance
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def download(root, study_id=None):
    reader = RangeReader()
    entries = archive_index(reader)
    source_csvs = {}
    for name in CSVS:
        report("source_csv", filename=name, bytes=entries[name]["size"])
        source_csvs[name] = read_member(reader, entries[name])
    tables = {name: csv_rows(raw) for name, raw in source_csvs.items()}
    study, series, coordinates, selected = choose_study(entries, tables[CSVS[0]], tables[CSVS[1]], study_id)
    report("selected_study", study_id=study, series=len(series), dicom_files=len(selected),
           dicom_bytes=sum(item["size"] for item in selected))
    files, csv_files = [], []
    train = [r for r in tables[CSVS[2]] if r["study_id"] == study]
    if len(train) != 1:
        raise ValueError("Selected study must have one training label row")
    for name, rows in zip(CSVS, (series, coordinates, train)):
        fields = next(csv.reader(io.StringIO(source_csvs[name].decode("utf-8-sig"))))
        raw = subset_csv(rows, fields)
        atomic_write(local_path(root, name), raw)
        csv_files.append({"relative_path": name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "zip_crc_verified": True, "subset": True, "source_crc32": entries[name]["crc"],
                      "source_sha256": hashlib.sha256(source_csvs[name]).hexdigest()})
    for ordinal, (item, raw) in enumerate(grouped_members(reader, selected), 1):
        report("dicom", completed=ordinal - 1, total=len(selected))
        atomic_write(local_path(root, item["name"]), raw)
        files.append({"relative_path": item["name"], "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "zip_crc_verified": True, "subset": False, "source_crc32": item["crc"]})
    provenance = {"repo_id": REPO, "revision": REVISION, "archive_url": URL,
                  "archive_size": ARCHIVE_SIZE, "archive_sha256": ARCHIVE_SHA256,
                  "whole_archive_sha256_verified": False, "source": SOURCE,
                  "license_note": "Original RSNA challenge terms; mirror supplies no license card",
                  "study_id": study, "requested_bytes_upper_bound": reader.total,
                  "series": [{"series_id": r["series_id"], "series_description": r["series_description"]} for r in series],
                  "method": "Bounded HTTP ranges; native DICOM bytes; per-member ZIP CRC32 and SHA256",
                  "files": files, "csv_files": csv_files}
    atomic_write(root / "provenance.json", (json.dumps(provenance, indent=2) + "\n").encode())
    report("download_complete", dicom_files=len(selected), requested_bytes_upper_bound=reader.total)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--study-id", help="Optional numeric RSNA study identifier")
    parser.add_argument("--verify-only", action="store_true", help="Verify local SHA256 without network")
    args = parser.parse_args(argv)
    try:
        root = args.output.resolve()
        outside_git(root)
        if args.study_id and not args.study_id.isdigit():
            raise ValueError("Study identifier must contain only digits")
        provenance = verify_existing(root)
        if provenance and args.study_id and provenance["study_id"] != args.study_id:
            raise ValueError("Existing study differs from requested identifier")
        if provenance is None:
            if args.verify_only:
                raise ValueError("Study missing, incomplete, or checksum verification failed")
            provenance = download(root, args.study_id)
        print(json.dumps({"status": "verified", "study_id": provenance["study_id"],
                          "files": len(provenance["files"]), "revision": REVISION,
                          "whole_archive_sha256_verified": False}))
        return 0
    except (OSError, ValueError, KeyError, struct.error, zlib.error) as error:
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
