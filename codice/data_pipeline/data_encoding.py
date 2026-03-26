import pandas as pd


class DataEncoding:
    """Gestisce l'encoding e la standardizzazione."""

    def __init__(self, dataframe: pd.DataFrame, scaler=None, is_train=True):
        self.df = dataframe.copy()
        self.scaler = scaler
        self.is_train = is_train

    def trasforma(self, lista_colonne: list = None) -> pd.DataFrame:
        """Esegue l'encoding e la standardizzazione."""
        self.dummy()
        self.standardizza(self.is_train)

        # Se in test set, allinea le colonne con il train set
        if not self.is_train and lista_colonne:
            self.df = self.df.reindex(columns=lista_colonne, fill_value=0)

        return self.df

    def dummy(self):
        """Trasforma le feature categoriche in dummy variables."""
        feature_categoriche = [
            'land_surface_condition', 'foundation_type', 'roof_type',
            'ground_floor_type', 'other_floor_type', 'position',
            'plan_configuration', 'legal_ownership_status'
        ]

        self.df = pd.get_dummies(
            self.df,
            columns=feature_categoriche,
            drop_first=False,
            dtype=int
        )

        nuove_colonne = [
            col for col in self.df.columns
            if any(feat in col for feat in feature_categoriche) and col not in feature_categoriche
        ]
        print(f"Aggiunte {len(nuove_colonne)} nuove colonne dummy.")

    def standardizza(self, is_train=True):
        """Standardizza le feature numeriche continue."""
        print("\n--- STANDARDIZZAZIONE ---")

        colonne_continue = [
            'age',
            'area_percentage',
            'height_percentage',
            'count_floors_pre_eq',
            'count_families'
        ]

        colonne_da_standardizzare = [col for col in colonne_continue if col in self.df.columns]

        if not colonne_da_standardizzare:
            print("Nessuna colonna continua trovata per la standardizzazione.")
            return

        if is_train:
            self.df[colonne_da_standardizzare] = self.scaler.fit_transform(
                self.df[colonne_da_standardizzare]
            )
            print(f"Scaler media di train: {self.scaler.mean_}")
            print(f"Standardizzazione calcolata e applicata (fit_transform) su: {colonne_da_standardizzare}")
        else:
            self.df[colonne_da_standardizzare] = self.scaler.transform(
                self.df[colonne_da_standardizzare]
            )
            print(f"Standardizzazione applicata (transform) su: {colonne_da_standardizzare}")

        print("\nEsempio di valori standardizzati (prime 5 righe):")
        print(self.df[colonne_da_standardizzare].head())
