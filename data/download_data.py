"""Download Batch1 and Batch2 from the MATR battery dataset release."""

from pathlib import Path
from urllib.request import Request, urlopen


DATASETS = {
    "Batch1": "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",
    "Batch2": "https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",
}

CHUNK_SIZE = 8 * 1024 * 1024


def download(url: str, destination: Path) -> None:
    if destination.exists():
        size_gb = destination.stat().st_size / 1024**3
        print(f"건너뜀: {destination.name}이 이미 있습니다 ({size_gb:.2f} GB).")
        return

    temporary = destination.with_suffix(".part")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    print(f"다운로드 시작: {destination.name}")
    try:
        with urlopen(request) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r  {downloaded / 1024**3:.2f} / {total / 1024**3:.2f} GB "
                        f"({downloaded / total * 100:.1f}%)",
                        end="",
                        flush=True,
                    )
        print()
        temporary.replace(destination)
        print(f"저장 완료: {destination}")
    except Exception:
        if temporary.exists():
            print(f"중단된 임시 파일: {temporary}")
        raise


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    for filename, url in DATASETS.items():
        download(url, data_dir / filename)


if __name__ == "__main__":
    main()
