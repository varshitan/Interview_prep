import os
import zipfile
import kaggle
import pandas as pd

# Configure Kaggle API credentials (optional if already set up)
os.environ["KAGGLE_USERNAME"] = "nelakurthivarshitha"
os.environ["KAGGLE_KEY"] = "e1cf88cee758ec7a4a788d5dd5ed46da"

# Kaggle dataset identifier
dataset = "ulrikthygepedersen/related-job-skills"

# Local file paths
zip_file = "related-job-skills.zip"
extract_to = "related-job-skills-data"

# Step 1: Download the dataset
print("Downloading dataset...")
kaggle.api.dataset_download_files(dataset, path=".", unzip=False)

# Step 2: Extract the dataset
print("Extracting dataset...")
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# Step 3: Load and explore the data
print("Loading dataset...")
csv_file = os.path.join(extract_to, "related_job_skills.csv")  # Update with actual file name
df = pd.read_csv(csv_file)

# Display the data
print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns.")
print(df.head())
