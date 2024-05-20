
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3


class VeriTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

train_dataset_fake = VeriTextDataset(train_fake_news_df['text'], train_fake_news_df['label'], tokenizer, MAX_LENGTH)
train_loader_fake = DataLoader(train_dataset_fake, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.train()
for epoch in range(NUM_EPOCHS):
    for batch in train_loader_fake:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()


test_dataset_fake = VeriTextDataset(test_fake_news_df['text'], test_fake_news_df['label'], tokenizer, MAX_LENGTH)
test_loader_fake = DataLoader(test_dataset_fake, batch_size=BATCH_SIZE)

model.eval()
y_true_fake = []
y_pred_fake = []

with torch.no_grad():
    for batch in test_loader_fake:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs.logits, dim=1)
        y_true_fake.extend(labels.cpu().numpy())
        y_pred_fake.extend(predicted_labels.cpu().numpy())


accuracy_labeled_fake = accuracy_score(y_true_fake, y_pred_fake)
print("Accuracy on labeled fake news test data:", accuracy_labeled_fake)


pseudo_labeling_df = labeled_real_news_df.sample(frac=0.2,abcnews. random_state=42)

pseudo_labels_real = [] 
for idx, row in pseudo_labeling_df.iterrows():
    input_text = row['headline_text']
    encoding = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted_label = torch.max(outputs.logits, dim=1)

    pseudo_labels_real.append(predicted_label.item())

pseudo_labeling_df['label'] = pseudo_labels_real

combined_train_df = pd.concat([train_fake_news_df, pseudo_labeling_df], ignore_index=True)

train_dataset_combined = VeriTextDataset(combined_train_df['text'], combined_train_df['label'], tokenizer, MAX_LENGTH)
train_loader_combined = DataLoader(train_dataset_combined, batch_size=BATCH_SIZE, shuffle=True)


model_finetune = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
optimizer_finetune = AdamW(model_finetune.parameters(), lr=LEARNING_RATE)

model_finetune.to(device)
model_finetune.train()
for epoch in range(NUM_EPOCHS):
    for batch in train_loader_combined:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer_finetune.zero_grad()
        outputs = model_finetune(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer_finetune.step()

test_dataset_real = VeriTextDataset(test_real_news_df['headline_text'], test_real_news_df['label'], tokenizer, MAX_LENGTH)
test_loader_real = DataLoader(test_dataset_real, batch_size=BATCH_SIZE)

model_finetune.eval()
y_true_real = []
y_pred_real = []

with torch.no_grad():
    for batch in test_loader_real:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model_finetune(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs.logits, dim=1)
        y_true_real.extend(labels.cpu().numpy())
        y_pred_real.extend(predicted_labels.cpu().numpy())


