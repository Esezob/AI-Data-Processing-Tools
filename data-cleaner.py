import pandas as pd

def clean_ai_training_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Remove duplicate entries to ensure data diversity
    df = df.drop_duplicates()
    
    # Handle missing values in critical annotation columns
    df = df.dropna(subset=['annotation_label', 'image_url'])
    
    # Standardize text format for consistency in NLP training
    df['text_description'] = df['text_description'].str.strip().str.lower()
    
    print(f"Data Cleaning Complete. Total records: {len(df)}")
    return df

# Example usage:
# clean_ai_training_data('raw_training_data.csv')
