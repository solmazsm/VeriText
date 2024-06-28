svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_fake_tfidf, y_train_fake)

X_test_fake_tfidf = tfidf_vectorizer.transform(X_test_fake)
y_pred_fake = svm_classifier.predict(X_test_fake_tfidf)
accuracy_fake = accuracy_score(y_test_fake, y_pred_fake)
print("Accuracy on labeled fake news test data:", accuracy_fake)
