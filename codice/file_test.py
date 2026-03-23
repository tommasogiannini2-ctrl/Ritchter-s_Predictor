import unittest

import unittest
import pandas as pd
from codice.data_reduction import DataReducer


class TestDataReducer(unittest.TestCase):

    def setUp(self):
        """
        Questo metodo viene eseguito in automatico prima di ogni test.
        Creazione di un dataset fittizio con 10.000 record e uno sbilanciamento noto:
        - Grado 1: 10% (1.000 record)
        - Grado 2: 70% (7.000 record)
        - Grado 3: 20% (2.000 record)
        """
        print("\n[SetUp] Creazione del dataset...")
        dati = {
            'building_id': range(10000),
            'dummy_feature': [1] * 10000,
            'damage_grade': [1] * 1000 + [2] * 7000 + [3] * 2000
        }
        self.df_fake = pd.DataFrame(dati)

    def test_stratificazione_mantenuta(self):
        """
        Testa se la riduzione mantiene inalterate le proporzioni della colonna target.
        """
        print("\n[Test] Esecuzione verifica stratificazione...")

        # Calcolo le proporzioni originali (da 0.0 a 1.0)
        prop_originali = self.df_fake['damage_grade'].value_counts(normalize=True)

        # Istanzio il tuo riduttore e calcolo la memoria
        reducer = DataReducer(self.df_fake)
        n_record, memoria_attuale = reducer.get_info()

        # Imposto un limite di memoria molto basso per forzare un taglio drastico (taglio del 90%)
        limite_mb_test = memoria_attuale / 10

        # Eseguo la riduzione
        df_ridotto = reducer.riduci_per_memoria(limite_mb_test)

        # Calcolo le proporzioni dopo il taglio
        prop_ridotte = df_ridotto['damage_grade'].value_counts(normalize=True)

        # VERIFICA MATEMATICA
        for grado in [1, 2, 3]:
            # assertAlmostEqual verifica che i numeri siano uguali fino a un certo numero di decimali.
            # Se la stratificazione fallisce, il test si blocca qui e lancia un errore.
            self.assertAlmostEqual(
                prop_originali[grado],
                prop_ridotte[grado],
                places=2,
                msg=f"Fallimento sul Grado {grado}: Proporzione alterata!"
            )

        print(f"Test superato: Il dataset è passato da {len(self.df_fake)} a {len(df_ridotto)} record.")
        print("Le proporzioni delle classi sono rimaste identiche.")


if __name__ == '__main__':
    unittest.main()