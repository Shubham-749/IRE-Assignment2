"""_unzip() must track completion per-zip, not by whether the shared extract
directory merely has *something* in it -- EB-NeRD's large-scale download extracts 3
different zips into one shared directory, and a real run silently skipped extracting
2 of them because the directory already had content from the first.
"""

import zipfile

from ire_a1.download import _unzip


def _make_zip(path, filename, content=b"data"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(filename, content)


def test_second_zip_into_shared_dir_is_not_silently_skipped(tmp_path):
    extract_dir = tmp_path / "shared"
    zip_a = tmp_path / "a.zip"
    zip_b = tmp_path / "b.zip"
    _make_zip(zip_a, "from_a.txt")
    _make_zip(zip_b, "from_b.txt")

    _unzip(zip_a, extract_dir)
    _unzip(zip_b, extract_dir)  # must not be skipped just because extract_dir is non-empty

    assert (extract_dir / "from_a.txt").exists()
    assert (extract_dir / "from_b.txt").exists()


def test_same_zip_extracted_twice_is_skipped_without_force(tmp_path):
    extract_dir = tmp_path / "shared"
    zip_a = tmp_path / "a.zip"
    _make_zip(zip_a, "from_a.txt")

    _unzip(zip_a, extract_dir)
    (extract_dir / "from_a.txt").write_text("modified after extraction")
    _unzip(zip_a, extract_dir)  # should skip -- marker already present

    assert (extract_dir / "from_a.txt").read_text() == "modified after extraction"


def test_force_re_extracts_even_with_marker_present(tmp_path):
    extract_dir = tmp_path / "shared"
    zip_a = tmp_path / "a.zip"
    _make_zip(zip_a, "from_a.txt", content=b"original")

    _unzip(zip_a, extract_dir)
    (extract_dir / "from_a.txt").write_bytes(b"modified")
    _unzip(zip_a, extract_dir, force=True)

    assert (extract_dir / "from_a.txt").read_bytes() == b"original"
