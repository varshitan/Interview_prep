import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Step 1: Load and Prepare the Dataset
# Replace 'your_dataset.csv' with your actual file path
data = pd.read_csv("Jobs/jobs.csv")

# Combine relevant columns into a single text input
data['input_text'] = "Skills: " + data['Key Skills'].apply(lambda x: x.replace("|", ", ")) + \
                     ". Experience: " + data['Job Experience Required']

# Encode target labels (Role Category)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Role Category'])

# Split the data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['input_text'], data['label'], test_size=0.2, random_state=42
)

# Step 2: Tokenize the Text Data
# Load the tokenizer for a pre-trained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the input text
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Step 3: Prepare Dataset for Transformers
class JobDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
# Ensure data is properly aligned after train-test split
train_texts = train_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)

# Create datasets
train_dataset = JobDataset(train_encodings, train_labels)
test_dataset = JobDataset(test_encodings, test_labels)

# Ensure labels are aligned
assert len(train_dataset) == len(train_labels), "Mismatch between dataset length and labels"


# Step 4: Load Pre-Trained Model
num_labels = len(label_encoder.classes_)  # Number of unique categories
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Step 5: Define Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save results
    num_train_epochs=3,             # Number of training epochs
    per_device_train_batch_size=16, # Batch size per device during training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    warmup_steps=500,               # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,              # Strength of weight decay
    logging_dir="./logs",           # Directory for logging
    evaluation_strategy="epoch",    # Evaluate at the end of each epoch
    save_strategy="epoch",          # Save model at the end of each epoch
)

trainer = Trainer(
    model=model,                     # The pre-trained model
    args=training_args,              # Training arguments
    train_dataset=train_dataset,     # Training dataset
    eval_dataset=test_dataset,       # Evaluation dataset
)

# Step 6: Train the Model
trainer.train()

# Step 7: Save the Model and Tokenizer
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Load the trained model for inference
model = AutoModelForSequenceClassification.from_pretrained("./trained_model")
tokenizer = AutoTokenizer.from_pretrained("./trained_model")

# Inference: Predict Role Category for New Text
new_job_description = "Skills: Python, Django, API development. Experience: 3 years."
inputs = tokenizer(new_job_description, truncation=True, padding=True, max_length=128, return_tensors="pt")
outputs = model(**inputs)

# Get predicted label
predicted_label = torch.argmax(outputs.logits, dim=1).item()
predicted_role = label_encoder.inverse_transform([predicted_label])[0]

print("Predicted Role Category:", predicted_role)
