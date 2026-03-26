import pandas as pd
from sklearn.preprocessing import StandardScaler

# Importa i moduli della pipeline
from data_cleaning import DataCleaning
from data_imputation import DataImputation
from data_encoding import DataEncoding



class Preprocessing:
    """
    Orchestratore principale della pipeline di preprocessing.
    Coordina pulizia, imputazione e encoding del dataset.
    """

    def __init__(self, dataframe: pd.DataFrame, scaler=None, lista_colonne=None, is_train=True):
        self.df = dataframe.copy()
        self.scaler = scaler if scaler else StandardScaler()
        self.lista_colonne = lista_colonne if lista_colonne else []
        self.is_train = is_train

    def esegui(self) -> pd.DataFrame:
        """Esegue l'intera pipeline di preprocessing."""
        print(f"\n{'=' * 60}")
        print(f"Avvio Preprocessing ({'Train' if self.is_train else 'Test'})...")
        print(f"{'=' * 60}")

        # FASE 1: PULIZIA
        print(f"\n[FASE 1/3] Pulizia dei dati...")
        cleaning = DataCleaning(self.df)

        if self.is_train:
            cleaning.elimina_record_null_percentuale()
            cleaning.elimina_colonne_nulle()

        self.df = cleaning.pulisci()

        # FASE 2: IMPUTAZIONE
        print(f"\n[FASE 2/3] Imputazione dei valori mancanti...")
        imputation = DataImputation(self.df, self.scaler, self.is_train)
        self.df = imputation.imputa()

        # FASE 3: ENCODING E STANDARDIZZAZIONE
        print(f"\n[FASE 3/3] Encoding e standardizzazione...")
        encoding = DataEncoding(self.df, self.scaler, self.is_train)
        self.df = encoding.trasforma(self.lista_colonne)

        # Aggiornamento della lista di colonne per il test set
        if self.is_train:
            self.lista_colonne = self.df.columns.tolist()

        print(f"\n{'=' * 60}")
        print(f"Preprocessing completato! Dataset shape: {self.df.shape}")
        print(f"{'=' * 60}\n")

        return self.df
