import pandas as pd
import os


data_dir = 'data'


import pandas as pd
import os

# Questo comando trova la cartella dove si trova fisicamente il file preprocessing.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Torniamo indietro di un livello (nella root del progetto) e puntiamo a 'data'
data_dir = os.path.join(current_dir, '..', 'data')

print(f"Sto cercando i dati in: {os.path.abspath(data_dir)}")

# Caricamento dei file CSV
try:
    train_values = pd.read_csv(os.path.join(data_dir, 'train_values.csv'))
    train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
    test_values = pd.read_csv(os.path.join(data_dir, 'test_values.csv'))
    print(" File caricati con successo!")
except FileNotFoundError as e:
    print(f" Errore: Non trovo i file. Controlla che la cartella 'data' sia corretta.")
    print(e)
    exit() # Ferma lo script se non trova i file

# Ora il merge funzionerà perché i nomi sono definiti
train_full = train_values.merge(train_labels, on='building_id')

# 4. Visualizzazione di controllo
print("\n--- INFO DATASET DI TRAINING ---")
print(f"Numero totale di righe (edifici): {train_full.shape[0]}")
print(f"Numero totale di colonne (feature): {train_full.shape[1]}")

print("\n--- PRIME RIGHE DEL TRAINING SET ---")
print(train_full.head())

print("\n--- DISTRIBUZIONE DEL TARGET (damage_grade) ---")
# Vediamo quanti edifici hanno danno 1, 2 o 3
print(train_full['damage_grade'].value_counts().sort_index())

# 5. Verifica valori mancanti
print("\n--- VALORI MANCANTI ---")
print(train_full.isnull().sum().sum())