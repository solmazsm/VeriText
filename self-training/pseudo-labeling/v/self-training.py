#self-training approach by pseudo-labeling the unlabeled


tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_fake_tfidf = tfidf_vectorizer.fit_transform(X_train_fake)


svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_fake_tfidf, y_train_fake)


X_test_fake_tfidf = tfidf_vectorizer.transform(X_test_fake)
y_pred_fake = svm_classifier.predict(X_test_fake_tfidf)
accuracy_fake = accuracy_score(y_test_fake, y_pred_fake)
print("Accuracy on labeled fake news test data:", accuracy_fake)


X_unlabeled_real = real_news_df['headline_text'][:1000]  
X_unlabeled_real_tfidf = tfidf_vectorizer.transform(X_unlabeled_real)
pseudo_labels_real = svm_classifier.predict(X_unlabeled_real_tfidf)

X_combined = pd.concat([X_train_fake, X_unlabeled_real])
y_combined = pd.concat([y_train_fake, pd.Series(pseudo_labels_real)])


X_combined_tfidf = tfidf_vectorizer.transform(X_combined)
svm_classifier.fit(X_combined_tfidf, y_combined)


