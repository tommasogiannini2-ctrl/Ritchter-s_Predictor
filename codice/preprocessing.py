import pandas as pd
import os
from abc import ABC, abstractmethod
import time
from sklearn.impute import KNNImputer


# --- 1. DEFINIZIONE DELLA FACTORY ---
class AbstractOpener(ABC):
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

# --- 2. CLASSE PREPROCESSING ---
class Preprocessing:
    """
    Classe incaricata della pulizia e preparazione del dataset.
    Riceve il dataframe già unito e restituisce il dataframe processato.
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def esegui(self) -> pd.DataFrame:
        """
        Punto di ingresso per tutte le operazioni di pulizia.
        """
        print("\nAvvio Preprocessing...")
        self.elimina_duplicati()

        self.rimuovi_outlier_strutturali()

        self.elimina_classnull()
        self.elimina_colonne_nulle()
        self.elimina_record_null_percentuale()

        self.gestisci_valori_mancanti()

        return self.df

    # Metodo per eliminare duplicati
    def elimina_duplicati(self):
        self.df = self.df.drop_duplicates()
        # Riassegna gli indici dopo l'eliminazione
        self.df = self.df.reset_index(drop=True)

    def gestisci_valori_mancanti(self):
        """Gestione interattiva dei valori nulli (NaN)."""
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
                dat.dropna(inplace=True)
                print("Righe con valori mancanti eliminate con successo!")

            elif choice == 2:
                # Imputazione con la media su tutte le colonne numeriche che hanno NaN
                numeric_cols_with_nan = dat.select_dtypes(include='number').columns[
                    dat.select_dtypes(include='number').isnull().any()
                ]
                for col in numeric_cols_with_nan:
                    col_mean = dat[col].mean()
                    dat[col].fillna(col_mean, inplace=True)
                print(f"Imputazione univariata eseguita su colonne: {list(numeric_cols_with_nan)}")

            elif choice == 3:
                numeric_cols = dat.select_dtypes(include='number').columns
                imputer = KNNImputer(n_neighbors=5)
                dat[numeric_cols] = imputer.fit_transform(dat[numeric_cols])
                print("Imputazione multivariata (KNN) eseguita con successo!")

            elif choice == 4:
                print("Uscita senza modifiche.")
                break

            # Ricalcola n dopo ogni operazione per decidere se uscire dal loop
            n = dat.isnull().sum().sum()
            if n == 0:
                print("Nessun valore mancante rimasto. Pulizia completata!")

        # Aggiornamento del dataframe
        self.df = dat


    def rimuovi_outlier_strutturali(self):
        """
        Rimuove record che non rispettano i domini di valore attesi.
        Utile per pulire errori di inserimento dati.
        """
        print("\n--- CONTROLLO OUTLIER ---")

        # Valori attesi per ogni colonna categorica/binaria
        valid_values = {
            'land_surface_condition': ['n', 'o', 't'],
            'foundation_type': ['h', 'i', 'r', 'u', 'w'],
            'roof_type': ['n', 'q', 'x'],
            'ground_floor_type': ['f', 'm', 'v', 'x', 'z'],
            'other_floor_type': ['j', 's', 'q', 'x'],
            'position': ['j', 's', 'o', 't'],
            'plan_configuration': ['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'],
            'legal_ownership_status': ['a', 'r', 'v', 'w'],
            'has_superstructure_adobe_mud': [0, 1],
            'has_superstructure_mud_mortar_stone': [0, 1],
            'has_superstructure_stone_flag': [0, 1],
            'has_superstructure_cement_mortar_stone': [0, 1],
            'has_superstructure_mud_mortar_brick': [0, 1],
            'has_superstructure_cement_mortar_brick': [0, 1],
            'has_superstructure_timber': [0, 1],
            'has_superstructure_rc_non_engineered': [0, 1],
            'has_superstructure_rc_engineered': [0, 1],
            'has_superstructure_other': [0, 1],
            'has_secondary_use': [0, 1],
            'has_secondary_use_agriculture': [0, 1],
            'has_secondary_use_hotel': [0, 1],
            'has_secondary_use_rental': [0, 1],
            'has_secondary_use_institution': [0, 1],
            'has_secondary_use_school': [0, 1],
            'has_secondary_use_industry': [0, 1],
            'has_secondary_use_health_post': [0, 1],
            'has_secondary_use_gov_office': [0, 1],
            'has_secondary_use_use_police': [0, 1],
            'has_secondary_use_other': [0, 1],
        }

        # Costruisci una maschera booleana: True = riga valida
        valid_mask = (
                self.df['geo_level_1_id'].between(0, 30) &
                self.df['geo_level_2_id'].between(0, 1427) &
                self.df['geo_level_3_id'].between(0, 12567)
        )

        for col, values in valid_values.items():
            if col in self.df.columns:
                valid_mask &= self.df[col].isin(values)

        righe_prima = len(self.df)
        self.df = self.df[valid_mask].reset_index(drop=True)
        righe_dopo = len(self.df)

        print(f"Righe rimosse come outlier: {righe_prima - righe_dopo}")
        print(f"Righe rimanenti: {righe_dopo}")

        print("\nPreprocessing completato!")


    def elimina_classnull(self):
        target_col = 'damage_grade'
        righe_originali = self.df.shape[0]
        # Rimuove le righe dove il valore nella colonna 'damage_grade' è nullo (NaN)
        self.df = self.df.dropna(subset=[target_col]).reset_index(drop=True)
        righe_dopo_aver_tolto_i_null = len(self.df)
        assert righe_dopo_aver_tolto_i_null <= righe_originali, "ERRORE: le righe dopo la pulizia sono aumentate!"

    def elimina_record_null_percentuale(self, soglia_percentuale=0.30):
        """
        Rimuove i record che hanno una percentuale di valori nulli superiore alla soglia.
        soglia_percentuale=0.30.
        """
        n_colonne = len(self.df.columns)
        min_valori_validi = int(n_colonne * (1 - soglia_percentuale))

        dati_puliti = self.df.dropna(thresh=min_valori_validi).reset_index(drop=True)

        righe_eliminate = len(dati) - len(dati_puliti)
        print(f"Record eliminati per troppi null sulla riga (Soglia {soglia_percentuale * 100}%): {righe_eliminate}")

    def elimina_colonne_nulle(self,soglia_percentuale=0.4):
        """
        Rimuove le feature che superano la percentuale di valori nulli indicata.
        Soglia 0.4 = Elimina se mancano più del 40% dei dati.
        """
        #  Calcolo del numero massimo di nulli consentiti per colonna
        n_righe_totali = len(self.df)
        limite_nulli = n_righe_totali * soglia_percentuale

        # Identificazione delle colonne che superano il limite
        serie_nulli = self.df.isnull().sum()
        colonne_da_eliminare = serie_nulli[serie_nulli > limite_nulli].index.tolist()

        # Log dei risultati (utile per la tua relazione FIA)
        if colonne_da_eliminare:
            print(
                f"Attenzione: Eliminate {len(colonne_da_eliminare)} feature (> {soglia_percentuale * 100}% nulli).")
            print(f"Feature rimosse: {colonne_da_eliminare}")
            # Restituiamo il dataframe senza quelle colonne
            self.df.drop(columns=colonne_da_eliminare)
        else:
            print(f"Controllo Qualità: Tutte le feature rispettano la soglia del {soglia_percentuale * 100}%.")



# --- 3. MAIN SCRIPT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')

try:
    path_values = os.path.join(data_dir, 'train_values.csv')
    path_labels = os.path.join(data_dir, 'train_labels.csv')

    # CARICAMENTO
    print("Caricamento file in corso...")
    train_values = scegli_opener(path_values).open(path_values)
    train_labels = scegli_opener(path_labels).open(path_labels)

    # MERGE
    dati = train_values.merge(train_labels, on='building_id')
    print(f"File caricati e uniti. Righe totali: {dati.shape[0]}")

    # PREPROCESSING
    preprocessor = Preprocessing(dati)
    df_processato = preprocessor.esegui()

    print("\n--- RESOCONTO FINALE ---")
    print(f"Dimensioni Righe: {df_processato.shape[0]}")
    print(f"Dimensioni Colonne:  {df_processato.shape[1]} \n")

    print(f"Informazioni sul dataframe:")
    df_processato.info()
    print(f"\nValori mancanti residui: {df_processato.isnull().sum().sum()}")

except Exception as ex:
    print(f"Errore: {ex}")

# 19 marzo righe totali e 0 null
# File caricati e uniti. Righe totali: 260601