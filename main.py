from roberta_model import textualize_flow, IoMTFlowDataset, compute_metrics_fn
from data_loader import load_iomt_data

# Data Directory
DATA_DIR = "/data/user/bsindala/PhD/Research/CICIoMT2024/WiFI and MQTT/attacks/CSV/" 

CLASS_CONFIG = 19 # Choose 19, 6, or 2 based on your experiment

# --- Main Execution ---
if __name__ == '__main__':
    # Load and preprocess data
    # This now returns DataFrames for X, which we'll iterate over for textualization
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, label_encoder, feature_columns_from_data = \
        load_iomt_data(DATA_DIR, CLASS_CONFIG)

    # Textualize the data
    # FEATURES_TO_TEXTUALIZE should now be feature_columns_from_data
    print("Textualizing training data...")
    train_texts = [textualize_flow(row, feature_columns_from_data) for _, row in X_train_df.iterrows()]
    print("Textualizing validation data...")
    val_texts = [textualize_flow(row, feature_columns_from_data) for _, row in X_val_df.iterrows()]
    print("Textualizing test data...")
    test_texts = [textualize_flow(row, feature_columns_from_data) for _, row in X_test_df.iterrows()]

#     # Initialize tokenizer and model
#     tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
#     num_labels = len(label_encoder.classes_)
#     model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

#     # Create datasets
#     train_dataset = IoMTFlowDataset(train_texts, y_train, tokenizer, MAX_SEQ_LENGTH)
#     val_dataset = IoMTFlowDataset(val_texts, y_val, tokenizer, MAX_SEQ_LENGTH)
#     test_dataset_for_eval = IoMTFlowDataset(test_texts, y_test, tokenizer, MAX_SEQ_LENGTH) # For final evaluation

#     # --- Handle Class Imbalance (Example: Weighted Loss) ---
#     # Calculate class weights (inverse frequency)
#     # This needs to be done on the y_train before it's turned into a dataset if possible, or passed to Trainer
#     class_counts = np.bincount(y_train) # y_train should be the integer encoded labels for the training set
#     class_weights = 1. / class_counts
#     class_weights = class_weights / np.sum(class_weights) # Normalize
#     class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(model.device if torch.cuda.is_available() else "cpu")
    
#     # Custom Trainer to handle weighted loss (if needed, or configure loss in TrainingArguments if possible)
#     # For PyTorch CrossEntropyLoss, weights are passed directly.
#     # The Hugging Face Trainer might require a custom `compute_loss` method if directly passing weights to `CrossEntropyLoss` is not straightforwardly supported for the Trainer's default loss.
#     # However, often the Trainer handles this well if the loss function inside the model (like RobertaForSequenceClassification) can accept weights,
#     # or by subclassing Trainer. Let's try simpler first.
#     # RobertaForSequenceClassification uses CrossEntropyLoss.
#     # We might need to modify the model's forward pass or use a custom Trainer.
#     # For now, we'll note this and proceed. If using a custom loop, you'd pass `weight=class_weights_tensor` to `nn.CrossEntropyLoss`.
    
#     print(f"Calculated class weights: {class_weights_tensor}")


#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         num_train_epochs=NUM_EPOCHS,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         learning_rate=LEARNING_RATE,
#         warmup_steps=500, # Number of warmup steps for learning rate scheduler
#         weight_decay=0.01,
#         logging_dir=LOGGING_DIR,
#         logging_steps=100, # Log every 100 steps
#         evaluation_strategy="epoch", # Evaluate at the end of each epoch
#         save_strategy="epoch",       # Save model at the end of each epoch
#         load_best_model_at_end=True, # Load the best model found during training
#         metric_for_best_model="f1",  # Use f1 score to determine the best model
#         report_to="tensorboard",     # Enable tensorboard logging
#         fp16=torch.cuda.is_available(), # Enable mixed precision training if CUDA is available
#     )

#     # Initialize Trainer
#     # To pass class_weights to the loss function used by RobertaForSequenceClassification,
#     # you might need to create a custom Trainer and override the compute_loss method.
#     class CustomTrainer(Trainer):
#         def compute_loss(self, model, inputs, return_outputs=False):
#             labels = inputs.pop("labels")
#             outputs = model(**inputs)
#             logits = outputs.get("logits")
#             # Ensure class_weights_tensor is on the same device as logits and labels
#             loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
#             loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#             return (loss, outputs) if return_outputs else loss

#     trainer = CustomTrainer( # Use CustomTrainer if you need weighted loss
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         compute_metrics=compute_metrics_fn,
#         tokenizer=tokenizer, # Useful for padding in DataCollatorWithPadding (default)
#     )
    
#     # --- Train the model ---
#     print("Starting RoBERTa fine-tuning...")
#     trainer.train()

#     # --- Evaluate the model ---
#     print("Evaluating on the test set...")
#     eval_results = trainer.evaluate(eval_dataset=test_dataset_for_eval)
#     print(f"Test set evaluation results: {eval_results}")

#     # --- Save the model and tokenizer ---
#     trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
#     tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model_tokenizer"))
#     label_encoder_path = os.path.join(OUTPUT_DIR, "final_model_label_encoder.npy")
#     np.save(label_encoder_path, label_encoder.classes_)
#     print(f"Model, tokenizer, and label encoder classes saved to {OUTPUT_DIR}")

    # --- Placeholder for Baseline Model Training & Evaluation ---
    # You would add functions here to train and evaluate traditional ML models
    # using X_train_df, y_train (integer encoded), X_val_df, y_val, X_test_df, y_test.
    # Remember to apply StandardScaler to X_train_df.values, X_val_df.values, X_test_df.values
    # for these models.

    # Example:
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.preprocessing import StandardScaler
    #
    # print("\nTraining Random Forest baseline...")
    # scaler = StandardScaler()
    # X_train_scaled_baseline = scaler.fit_transform(X_train_df.fillna(0)) # fillna(0) or other imputation
    # X_test_scaled_baseline = scaler.transform(X_test_df.fillna(0))
    #
    # rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # rf_model.fit(X_train_scaled_baseline, y_train) # y_train is integer encoded
    # y_pred_rf = rf_model.predict(X_test_scaled_baseline)
    #
    # rf_accuracy = accuracy_score(y_test, y_pred_rf)
    # rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted')
    # print(f"Random Forest - Test Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}, Precision: {rf_precision:.4f}, Recall: {rf_recall:.4f}")