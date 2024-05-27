
abcnews_dataset = VeriDataset(real_news_df['headline_text'], [1] * len(real_news_df), tokenizer, MAX_LENGTH)
abcnews_loader = DataLoader(abcnews_dataset, batch_size=BATCH_SIZE)

predictions = []
fine_tuned_model.eval()
with torch.no_grad():
    for batch in abcnews_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = fine_tuned_model(input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs.logits, dim=1)
        predictions.extend(predicted_labels.cpu().numpy())

