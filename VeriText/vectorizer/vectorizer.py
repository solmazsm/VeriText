tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_fake_tfidf = tfidf_vectorizer.fit_transform(X_train_fake)

X_test_fake_tfidf = tfidf_vectorizer.transform(X_test_fake)


X_unlabeled_real = real_news_df['headline_text'][:1000]  
X_unlabeled_real_tfidf = tfidf_vectorizer.transform(X_unlabeled_real)
