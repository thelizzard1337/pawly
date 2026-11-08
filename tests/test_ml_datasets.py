"""Dataset preparation safety and split invariants (synthetic inputs, no downloads)."""

import io
import tarfile

import pytest

pytest.importorskip("numpy")

from app.ml.datasets import assign_splits, near_duplicate_groups, safe_extract  # noqa: E402


def _tar(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_safe_extract_rejects_traversal_and_filters_prefixes(tmp_path):
    good = tmp_path / "good.tar.gz"
    _tar(good, [("images/a.jpg", b"1"), ("other/b.txt", b"2")])
    assert safe_extract(good, tmp_path / "out", ("images/",)) == 1
    assert (tmp_path / "out/images/a.jpg").read_bytes() == b"1"
    assert not (tmp_path / "out/other").exists()
    assert safe_extract(good, tmp_path / "out", ("images/",)) == 1  # repeat-safe
    evil = tmp_path / "evil.tar.gz"
    _tar(evil, [("images/../../escape.jpg", b"x")])
    with pytest.raises(SystemExit):
        safe_extract(evil, tmp_path / "out2", ("images/",))
    assert not (tmp_path / "escape.jpg").exists()
    link = tmp_path / "link.tar.gz"
    with tarfile.open(link, "w:gz") as tar:
        info = tarfile.TarInfo("images/link.jpg")
        info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(info)
    assert safe_extract(link, tmp_path / "out3", ("images/",)) == 0  # links are skipped


def test_near_duplicates_need_both_hashes():
    # 0 and 1 are close in both hashes; 2 is close horizontally only (no chaining).
    groups = near_duplicate_groups([0, 1, 3], [0, 1, 0xFFFF], 2)
    assert groups[0] == groups[1] != groups[2]


def _sample(i, breed, official, species="dog", group=None):
    return {
        "sample_id": f"s{i}",
        "breed": breed,
        "species": species,
        "official_split": official,
        "group_id": group or f"s{i}",
    }


def test_splits_keep_groups_together_and_hold_out_breeds():
    samples = []
    breeds = [f"b{k}" for k in range(8)]
    for k, breed in enumerate(breeds):
        for j in range(10):
            i = k * 100 + j
            samples.append(_sample(i, breed, "test" if j < 4 else "trainval"))
    # A near-duplicate group spanning official trainval and test stays in one split.
    samples[5]["group_id"] = samples[0]["group_id"]
    samples.append(_sample(999, "cat", "test", species="cat"))
    known, val_unknown, test_unknown = assign_splits(
        samples, seed=0, holdout_breeds=4, val_fraction=0.2
    )
    assert len(known) == 4 and len(val_unknown) == len(test_unknown) == 2
    assert not set(val_unknown) & set(test_unknown)
    by_group = {}
    for s in samples:
        if s["species"] == "dog":
            by_group.setdefault(s["group_id"], set()).add(s["split"])
    assert all(len(v) == 1 for v in by_group.values())
    for s in samples:
        if s["species"] == "cat":
            assert s["split"] == "gate_negative" and s["label_status"] == "non_dog"
        elif s["breed"] in test_unknown:
            assert s["split"] == "test" and s["label_status"] == "heldout_unknown"
        elif s["breed"] in val_unknown:
            assert s["split"] == "val" and s["label_status"] == "heldout_unknown"
        else:
            assert s["label_status"] == "known"
    assert not any(s["split"] == "train" and s["label_status"] != "known" for s in samples)
