for batch in test_loader_fake:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    probs_fake.extend(probabilities[:, 1].cpu().numpy())


for batch in test_loader_real:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    probs_real.extend(probabilities[:, 1].cpu().numpy())

fpr_fake, tpr_fake, thresholds_fake = roc_curve(y_true_fake, probs_fake)

fpr_real, tpr_real, thresholds_real = roc_curve(y_true_real, probs_real)

auc_fake = roc_auc_score(y_true_fake, probs_fake)


auc_real = roc_auc_score(y_true_real, probs_real)
