import pandas as pd

def extract_unique_classes(training_csv_path):
    df = pd.read_csv(training_csv_path)

    classes = set()
    for row in df['Predicted'].dropna():
        for instance in row.split(';'):
            parts = instance.strip().split()
            if len(parts) >= 6:
                cat_id = int(parts[0])
                classes.add(cat_id)
    
    class_list = sorted(list(classes))
    print(f'Clases encontradas ({len(class_list)}): {class_list}')
    return class_list

training_csv_path = 'training.csv'
extract_unique_classes(training_csv_path)
