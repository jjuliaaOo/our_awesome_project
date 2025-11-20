epd_sequences = load_epd()
random_genomic_sequences = generate_negative_set()

#Очистка:
all_sequences = combine_and_remove_duplicates(epd_sequences, random_genomic_sequences)
padded_sequences = pad_sequences_to_same_length(all_sequences)

#Извлечение признаков (здесь будет несколько путей):

#Путь A (Традиционный): 
X_kmer = calculate_kmers_frequency(padded_sequences, k=3)

#Путь B (Современный): 
X_embeddings = dnabert_model.embed(padded_sequences)

#Линейные модели (Бейзлайн):

#Разделили данные на тренировочные и тестовые.
svm_model.fit(X_train_embeddings, y_train)
baseline_accuracy = svm_model.score(X_test_embeddings, y_test)

#Вывод: "SVM на эмбеддингах DNABERT показала точность 88%".

#Нелинейные модели:
rf_model = RandomForestClassifier().fit(X_train_embeddings, y_train)
mlp_model = MLPClassifier().fit(X_train_embeddings, y_train)
