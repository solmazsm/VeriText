X_combined = pd.concat([X_train_fake, X_unlabeled_real])
y_combined = pd.concat([y_train_fake, pd.Series(pseudo_labels_real)])
