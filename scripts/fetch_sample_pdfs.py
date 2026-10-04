"""Download a test corpus of ~20 public PDFs (manuals, handbooks, standards) into corpus/samples/.

    uv run python scripts/fetch_sample_pdfs.py          # download missing files
    uv run python scripts/fetch_sample_pdfs.py --list   # show what would be downloaded

All are freely published by their authors (public domain, Creative Commons, or vendor docs
offered for download). They are fetched locally and never committed: corpus/ is gitignored.
Mix: screenshot-heavy software manuals, figure-heavy handbooks, dense reference manuals,
several over 1000 pages. Total download is roughly 500 MB.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "corpus" / "samples"

# (subfolder/filename, url)
PDFS = [
    # Software manuals with lots of screenshots
    ("software/qgis-3.44-desktop-user-guide.pdf", "https://docs.qgis.org/3.44/pdf/en/QGIS-3.44-DesktopUserGuide-en.pdf"),
    ("software/qgis-3.44-training-manual.pdf", "https://docs.qgis.org/3.44/pdf/en/QGIS-3.44-TrainingManual-en.pdf"),
    ("software/qgis-3.44-gentle-gis-introduction.pdf", "https://docs.qgis.org/3.44/pdf/en/QGIS-3.44-GentleGISIntroduction-en.pdf"),
    ("software/libreoffice-25.2-writer-guide.pdf", "https://books.libreoffice.org/en/WG252/WG252-WriterGuide.pdf"),
    ("software/nextcloud-user-manual.pdf", "https://docs.nextcloud.com/server/latest/Nextcloud_User_Manual.pdf"),
    ("software/nextcloud-admin-manual.pdf", "https://docs.nextcloud.com/server/latest/Nextcloud_Server_Administration_Manual.pdf"),
    ("software/kicad-8-pcb-editor.pdf", "https://docs.kicad.org/8.0/en/pcbnew/pcbnew.pdf"),
    ("software/kicad-8-schematic-editor.pdf", "https://docs.kicad.org/8.0/en/eeschema/eeschema.pdf"),
    ("software/proxmox-ve-admin-guide.pdf", "https://pve.proxmox.com/pve-docs/pve-admin-guide.pdf"),
    ("software/pro-git-2nd-edition.pdf", "https://github.com/progit/progit2/releases/download/2.1.450/progit.pdf"),
    # Cloud / developer reference (1000+ pages)
    ("cloud/aws-s3-user-guide.pdf", "https://docs.aws.amazon.com/pdfs/AmazonS3/latest/userguide/s3-userguide.pdf"),
    ("cloud/aws-ec2-user-guide.pdf", "https://docs.aws.amazon.com/pdfs/AWSEC2/latest/UserGuide/ec2-ug.pdf"),
    ("cloud/aws-well-architected-framework.pdf", "https://docs.aws.amazon.com/pdfs/wellarchitected/latest/framework/wellarchitected-framework.pdf"),
    ("dev/postgresql-17-manual.pdf", "https://www.postgresql.org/files/documentation/pdf/17/postgresql-17-A4.pdf"),
    ("dev/kotlin-language-reference.pdf", "https://kotlinlang.org/docs/kotlin-reference.pdf"),
    # Handbooks, standards, government publications (figures, tables)
    ("handbooks/faa-pilots-handbook-of-aeronautical-knowledge.pdf", "https://www.faa.gov/sites/faa.gov/files/2022-03/pilot_handbook.pdf"),
    ("handbooks/nasa-systems-engineering-handbook.pdf", "https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf"),
    ("handbooks/irs-publication-17-your-federal-income-tax.pdf", "https://www.irs.gov/pub/irs-pdf/p17.pdf"),
    ("security/nist-cybersecurity-framework-2.0.pdf", "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf"),
    ("security/owasp-web-security-testing-guide-4.2.pdf", "https://github.com/OWASP/wstg/releases/download/v4.2/wstg-v4.2.pdf"),
]


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.7.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    with open(tmp, "rb") as f:
        if f.read(5) != b"%PDF-":
            tmp.unlink()
            raise ValueError("response is not a PDF")
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list files and exit")
    args = parser.parse_args()

    failed = 0
    for i, (name, url) in enumerate(PDFS, 1):
        dest = DEST / name
        if args.list or dest.exists():
            print(f"[{i:2}/{len(PDFS)}] {'exists ' if dest.exists() else 'pending'} {name}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i:2}/{len(PDFS)}] downloading {name} ...", end=" ", flush=True)
        try:
            download(url, dest)
            print(f"{dest.stat().st_size / 1e6:.1f} MB")
        except Exception as exc:
            failed += 1
            print(f"FAILED ({exc})")
    if not args.list:
        print(f"\nDone. Files are in {DEST}. Run `uv run ingest` to index them.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
