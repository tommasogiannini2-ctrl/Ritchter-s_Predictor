import pandas as pd
import time
from sklearn.impute import KNNImputer


class DataImputation:
    """Gestisce l'imputazione dei valori mancanti."""

    def __init__(self, dataframe: pd.DataFrame, scaler=None, is_train=True):
        self.df = dataframe.copy()
        self.scaler = scaler
        self.is_train = is_train

    def imputa(self) -> pd.DataFrame:
        """Esegue l'imputazione interattiva dei valori mancanti."""
        n = self.df.isnull().sum().sum()
        print(f"\nSono stati trovati {n} valori mancanti.")

        dat = self.df.copy()

        while n > 0:
            print(f"\nDato che sono stati trovati {n} valori mancanti, scegli un'operazione di pulizia:")
            print("1. Eliminazione del record")
            print("2. Imputazione univariata (media per ogni colonna numerica)")
            print("3. Imputazione multivariata (KNN Imputer)")
            print("4. Esci senza modifiche")

            try:
                choice = int(input("Inserisci la tua scelta (1-4): "))
            except ValueError:
                print("Scelta non valida. Inserisci un numero tra 1 e 4.")
                time.sleep(1)
                continue

            if choice not in [1, 2, 3, 4]:
                print("Scelta non valida. Riprova.")
                time.sleep(1)
                continue

            if choice == 1:
                dat = self.gestisci_valori_mancanti_rimozione(dat)
            elif choice == 2:
                dat = self.gestisci_valori_mancanti_media(dat)
            elif choice == 3:
                dat = self.gestisci_valori_mancanti_KNN(dat)
            elif choice == 4:
                print("Uscita senza modifiche.")
                break

            n = dat.isnull().sum().sum()
            if n == 0:
                print("Nessun valore mancante rimasto. Pulizia completata!")

        self.df = dat
        return self.df

    def gestisci_valori_mancanti_rimozione(self, da: pd.DataFrame) -> pd.DataFrame:
        """Rimuove le righe con valori mancanti."""
        da.dropna(inplace=True)
        da.reset_index(drop=True, inplace=True)
        print("Righe con valori mancanti eliminate con successo!")
        return da

    def gestisci_valori_mancanti_media(self, da: pd.DataFrame) -> pd.DataFrame:
        """Imputazione univariata con la media."""
        numeric_cols_with_nan = da.select_dtypes(include='number').columns[
            da.select_dtypes(include='number').isnull().any()]

        if self.is_train:
            for col in numeric_cols_with_nan:
                col_mean = da[col].mean()
                da[col] = da[col].fillna(col_mean)
            print(f"Imputazione univariata eseguita su colonne di train: {list(numeric_cols_with_nan)}")
        else:
            i = 0
            for col in numeric_cols_with_nan:
                col_mean = self.scaler.mean_[i]
                da[col] = da[col].fillna(col_mean)
                i += 1
            print(f"Imputazione univariata eseguita su colonne di test: {list(numeric_cols_with_nan)}")

        return da

    def gestisci_valori_mancanti_KNN(self, da: pd.DataFrame) -> pd.DataFrame:
        """Imputazione multivariata con KNN."""
        numeric_cols = da.select_dtypes(include='number').columns
        imputer = KNNImputer(n_neighbors=5)
        da[numeric_cols] = imputer.fit_transform(da[numeric_cols])
        print("Imputazione multivariata (KNN) eseguita con successo!")
        return da
