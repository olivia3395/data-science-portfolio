from __future__ import annotations

import shutil
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _copy_csvs(src_dir: Path, dst_dir: Path) -> bool:
    copied = 0
    for csv_file in src_dir.rglob('*.csv'):
        target = dst_dir / csv_file.name
        if not target.exists():
            shutil.copy2(csv_file, target)
        copied += 1
    return copied > 0


def download_with_kagglehub_competition(competition: str, raw_dir: Path) -> bool:
    try:
        import kagglehub
        logger.info('Trying kagglehub competition download: %s', competition)
        path = kagglehub.competition_download(competition)
        return _copy_csvs(Path(path), raw_dir)
    except Exception as exc:
        logger.warning('kagglehub competition download failed: %s', exc)
        return False


def download_with_kagglehub_dataset(dataset: str, raw_dir: Path) -> bool:
    try:
        import kagglehub
        logger.info('Trying kagglehub dataset download: %s', dataset)
        path = kagglehub.dataset_download(dataset)
        return _copy_csvs(Path(path), raw_dir)
    except Exception as exc:
        logger.warning('kagglehub dataset download failed: %s', exc)
        return False


def download_with_kaggle_api_competition(competition: str, raw_dir: Path) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        logger.info('Trying Kaggle API competition download: %s', competition)
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(competition, path=str(raw_dir), quiet=False)
        for zip_file in raw_dir.glob('*.zip'):
            shutil.unpack_archive(str(zip_file), str(raw_dir))
        return any(raw_dir.glob('*.csv'))
    except Exception as exc:
        logger.warning('Kaggle API competition download failed: %s', exc)
        return False


def download_with_kaggle_api_dataset(dataset: str, raw_dir: Path) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        logger.info('Trying Kaggle API dataset download: %s', dataset)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=str(raw_dir), unzip=True, quiet=False)
        return any(raw_dir.glob('*.csv'))
    except Exception as exc:
        logger.warning('Kaggle API dataset download failed: %s', exc)
        return False


def ensure_rossmann_data(raw_dir: str | Path, competition: str, dataset_fallback: str) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    required = {'train.csv', 'store.csv'}
    if required.issubset({p.name for p in raw_dir.glob('*.csv')}):
        logger.info('Rossmann raw data already present.')
        return raw_dir

    success = (
        download_with_kagglehub_competition(competition, raw_dir)
        or download_with_kaggle_api_competition(competition, raw_dir)
        or download_with_kagglehub_dataset(dataset_fallback, raw_dir)
        or download_with_kaggle_api_dataset(dataset_fallback, raw_dir)
    )
    if not success:
        raise RuntimeError(
            'Unable to download Rossmann data. Confirm Kaggle credentials and, for competition downloads, ensure you have accepted competition rules.'
        )
    return raw_dir
