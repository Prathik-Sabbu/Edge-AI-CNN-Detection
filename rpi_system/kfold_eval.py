"""
Stratified K-Fold Cross-Validation for Edge AI Animal Classifier.
Evaluates model stability and generalization across K independent folds,
reporting Mean Accuracy +/- Std Dev and Mean F1-Score +/- Std Dev.
"""
import sys
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add rpi_system to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config

def run_kfold(k_folds: int = 5, max_samples_per_class: int = 400, epochs_per_fold: int = 10):
    print("================================================================================")
    print(f"      STRATIFIED {k_folds}-FOLD CROSS-VALIDATION (STATISTICAL RIGOR CHECK)      ")
    print("================================================================================")

    data_dirs = [config.TRAIN_DIR, config.VAL_DIR]
    labels = ["cat", "dog", "elephant", "horse", "lion"]
    
    file_paths = []
    labels_list = []

    print(f"Gathering images across classes: {labels}...")
    for label_idx, label in enumerate(labels):
        class_files = []
        for d in data_dirs:
            p = d / label
            if p.exists():
                class_files.extend(list(p.glob("*.jpg")) + list(p.glob("*.jpeg")) + list(p.glob("*.png")))
        
        # Sample evenly per class for balanced, rapid K-Fold validation
        if max_samples_per_class and len(class_files) > max_samples_per_class:
            np.random.seed(42)
            class_files = list(np.random.choice(class_files, size=max_samples_per_class, replace=False))
        
        for f in class_files:
            file_paths.append(str(f))
            labels_list.append(label_idx)

    file_paths = np.array(file_paths)
    labels_list = np.array(labels_list)
    total_samples = len(file_paths)
    num_classes = len(labels)

    print(f"Total balanced dataset for K-Fold: {total_samples} images ({total_samples // num_classes} per class)")

    # 1. Feature Extraction Backbone (MobileNetV3-Small frozen)
    print("\n[Step 1/2] Precomputing MobileNetV3 feature embeddings for all samples...")
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(config.INPUT_WIDTH, config.INPUT_HEIGHT, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    # Batch extraction for maximum GPU/CPU speed
    batch_size = 64
    features = []
    t0 = time.time()
    
    import cv2
    for i in range(0, total_samples, batch_size):
        batch_paths = file_paths[i:i + batch_size]
        batch_imgs = []
        for fp in batch_paths:
            img = cv2.imread(fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (config.INPUT_WIDTH, config.INPUT_HEIGHT))
            batch_imgs.append(img)
        batch_tensor = np.array(batch_imgs, dtype=np.float32)
        batch_tensor = preprocess_input(batch_tensor)
        feats = base_model(batch_tensor, training=False).numpy()
        features.append(feats)
        if (i // batch_size) % 5 == 0 or i + batch_size >= total_samples:
            print(f"  Processed {min(i + batch_size, total_samples)} / {total_samples} images...")

    features = np.vstack(features)
    print(f"Feature embeddings extracted in {time.time() - t0:.1f}s. Shape: {features.shape}")

    # 2. Stratified K-Fold Execution
    print(f"\n[Step 2/2] Running Stratified {k_folds}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_f1_scores = []
    fold_precisions = []
    fold_recalls = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels_list), 1):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels_list[train_idx], labels_list[val_idx]

        # Classification Head
        inputs = tf.keras.Input(shape=(features.shape[1],))
        x = tf.keras.layers.Dropout(0.2)(inputs)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        
        clf_model = tf.keras.Model(inputs, outputs)
        clf_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        clf_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_per_fold,
            batch_size=32,
            verbose=0,
        )

        preds = np.argmax(clf_model.predict(X_val, verbose=0), axis=1)

        acc = accuracy_score(y_val, preds) * 100.0
        f1 = f1_score(y_val, preds, average="macro") * 100.0
        prec = precision_score(y_val, preds, average="macro", zero_division=0) * 100.0
        rec = recall_score(y_val, preds, average="macro", zero_division=0) * 100.0

        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        fold_precisions.append(prec)
        fold_recalls.append(rec)

        print(f"  • Fold {fold}/{k_folds}: Accuracy = {acc:.2f}% | Macro F1 = {f1:.2f}% | Precision = {prec:.2f}% | Recall = {rec:.2f}%")

    # 3. Aggregate Statistical Summary
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)

    print("\n================================================================================")
    print("                    FINAL K-FOLD STATISTICAL RESULTS                            ")
    print("================================================================================")
    print(f"  • Mean Cross-Validation Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  • Mean Macro F1-Score            : {mean_f1:.2f}% ± {std_f1:.2f}%")
    print(f"  • Lowest Fold / Highest Fold     : {min(fold_accuracies):.2f}% / {max(fold_accuracies):.2f}%")
    print("================================================================================")
    print(f"\nPublication & Resume Quote:")
    print(f"\"Validated model generalization using Stratified {k_folds}-Fold Cross-Validation,")
    print(f" achieving a mean accuracy of {mean_acc:.1f}% ± {std_acc:.1f}% ({mean_f1:.1f}% ± {std_f1:.1f}% Macro F1)\")")
    print("================================================================================\n")

if __name__ == "__main__":
    run_kfold()
