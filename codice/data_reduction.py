import pandas as pd
from sklearn.model_selection import train_test_split


class DataReducer:
    """
    Classe per la gestione della pesantezza del dataset.
    Permette di campionare i dati in base al numero di record o all'occupazione di memoria, mantenendo le percentuali del target costanti.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.target = 'damage_grade'  # Colonna per la stratificazione

    def get_info(self):
        """Restituisce numero di record e memoria occupata in MB."""
        n_record = len(self.df)
        # memory_usage(deep=True) analizza l'effettivo consumo della RAM
        memoria_mb = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
        return n_record, memoria_mb

    def riduci_per_memoria(self, limite_mb: float):
        """
        Calcola la frazione necessaria per far rientrare il dataset nel limite di MB
        e applica il campionamento stratificato.
        """
        n_record, memoria_attuale = self.get_info()

        if memoria_attuale <= limite_mb:
            print(f"Il dataset occupa già {memoria_attuale:.2f} MB. Nessuna riduzione necessaria.")
            return self.df

        # Calcolo della proporzione (es. se ho 100MB e ne voglio 40, tengo lo 0.4)
        frazione = limite_mb / memoria_attuale
        print(f"Riduzione necessaria: tengo il {frazione * 100:.1f}% dei dati per rientrare in {limite_mb} MB.")

        # Campionamento stratificato
        df_ridotto, _ = train_test_split(
            self.df,
            train_size=frazione,
            stratify=self.df[self.target] if self.target in self.df.columns else None,
            random_state=42
        )

        self.df = df_ridotto.reset_index(drop=True)
        return self.df

    def interfaccia_utente(self):
        """Gestisce il dialogo con l'utente per la riduzione."""
        n, mem = self.get_info()
        print("\n--- ANALISI DIMENSIONI DATASET ---")
        print(f"Record attuali: {n}")
        print(f"Memoria occupata: {mem:.2f} MB")

        scelta = input("\nLa dimensione del dataset è ottimale? (s/n): ").lower()

        if scelta == 'n':
            try:
                limite = float(input("Quanti MB al massimo deve occupare il dataset? "))
                self.df = self.riduci_per_memoria(limite)
                n_nuovo, mem_nuova = self.get_info()
                print(f"Nuove dimensioni: {n_nuovo} record ({mem_nuova:.2f} MB)")
            except ValueError:
                print("Valore non valido. Procedo senza riduzioni.")

        return self.df