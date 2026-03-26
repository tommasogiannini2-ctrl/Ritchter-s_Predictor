import pandas as pd
import os
from abc import ABC, abstractmethod


class AbstractOpener(ABC):
    """Interfaccia astratta per l'apertura di file."""

    def open(self, dataframe_path: str) -> pd.DataFrame:
        if not os.path.exists(dataframe_path):
            raise FileNotFoundError(f"File {dataframe_path} non trovato")
        try:
            return self._load_data(dataframe_path)
        except Exception as e:
            raise RuntimeError(f"Errore durante la lettura del file {dataframe_path}: {e}")

    @abstractmethod
    def _load_data(self, path: str) -> pd.DataFrame:
        pass


class XLSOpener(AbstractOpener):
    def _load_data(self, path: str) -> pd.DataFrame:
        return pd.read_excel(path)


class CSVOpener(AbstractOpener):
    def _load_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)


class JSONOpener(AbstractOpener):
    def _load_data(self, path: str) -> pd.DataFrame:
        return pd.read_json(path)


def scegli_opener(dataframe_path: str) -> AbstractOpener:
    """Factory function per selezionare l'opener corretto."""
    ext = dataframe_path.split('.')[-1].lower()
    match ext:
        case 'csv' | 'txt':
            return CSVOpener()
        case 'xls' | 'xlsx':
            return XLSOpener()
        case 'json':
            return JSONOpener()
        case _:
            raise RuntimeError(f"Tipo di file non supportato: {ext}")
