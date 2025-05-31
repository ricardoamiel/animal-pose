import pandas as pd
import os

def limpiar_training(training_csv, eval_csv):
    training_df = pd.read_csv(training_csv)
    eval_df = pd.read_csv(eval_csv)
    
    eval_ids = set(eval_df['Id'].dropna().astype(int))
    print(f'IDs en eval: {sorted(eval_ids)}')

    # Filtrar training: solo IDs que NO estén en eval
    cleaned_training_df = training_df[~training_df['Id'].isin(eval_ids)]

    # Guardar el DataFrame limpio
    cleaned_training_df.to_csv('training_cleaned.csv', index=False)
    
    print(f'Filas originales en training: {len(training_df)}')
    print(f'Filas luego de limpieza: {len(cleaned_training_df)}')

    return cleaned_training_df

TRAINING_CSV = os.path.join("training.csv")
EVAL_CSV = os.path.join("eval.csv")
limpiar_training(TRAINING_CSV, EVAL_CSV)