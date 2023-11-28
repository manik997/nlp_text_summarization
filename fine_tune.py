import torch
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, pipeline
from torch.utils.data import Dataset
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv('/Users/manikmalhotra/Downloads/news_dataa.csv', delimiter='\t')
df = df[['text', 'label']]

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Save the label_encoder using joblib
joblib.dump(label_encoder, '/Users/manikmalhotra/Downloads/document_summarization/fine_tuned_model/label_encoder.joblib')

# Create a label mapping
label_mapping = dict(zip(df['label'], label_encoder.classes_))

# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=20)

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['text']
        self.targets = dataframe['label']
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        target = int(self.targets.iloc[index])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(target, dtype=torch.long),
        }

# Tokenizer and model initialization
tokenizer_classification = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_classification = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_))

# Save the tokenizer to the specified directory
tokenizer_classification.save_pretrained('/Users/manikmalhotra/Downloads/document_summarization/fine_tuned_model/tokenizer')

# Dataset initialization
train_dataset = CustomDataset(train_df, tokenizer_classification, max_len=512)
val_dataset = CustomDataset(val_df, tokenizer_classification, max_len=512)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Training setup
trainer = Trainer(
    model=model_classification,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=None,  
)

# Optimizer
optimizer = AdamW(model_classification.parameters(), lr=5e-5)

# Training loop
for epoch in range(training_args.num_train_epochs):
    for step, batch in enumerate(trainer.get_train_dataloader()):
        model_classification.train()
        batch = {k: v.to(model_classification.device) for k, v in batch.items()}
        outputs = model_classification(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        

    # Validation loop
    model_classification.eval()
    for batch in trainer.get_eval_dataloader():
        with torch.no_grad():
            batch = {k: v.to(model_classification.device) for k, v in batch.items()}
            outputs = model_classification(**batch)
            loss = outputs.loss

            

# Save the fine-tuned model
model_classification.save_pretrained('/Users/manikmalhotra/Downloads/document_summarization/fine_tuned_model')



