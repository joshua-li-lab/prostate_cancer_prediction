import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Layer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Create results directory
os.makedirs('results_folder', exist_ok=True)

# Parameters for N and cutoff
Ns = [4, 5, 6, 7, 8, 9]
cutoffs = [4, 5, 6, 7, 8, 9]

# Machine learning hyperparameters
t2v_kernel_size = 3
lstm_units = 128
epochs = 20
batch_size = 32
validation_split = 0.1
random_state = 42
loss = 'binary_crossentropy'
k_folds = 5
early_stop_patience = 5

class Time2Vec(Layer):
    def __init__(self, kernel_size=64, **kwargs):
        self.k = kernel_size
        super(Time2Vec, self).__init__(**kwargs)

    def build(self, input_shape):
        self.wb = self.add_weight(name='wb', shape=(1,), initializer='uniform', trainable=True)
        self.bb = self.add_weight(name='bb', shape=(1,), initializer='uniform', trainable=True)
        self.wa = self.add_weight(name='wa', shape=(1, self.k), initializer='uniform', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(1, self.k), initializer='uniform', trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs):
        bias = self.wb * inputs + self.bb
        dp = tf.keras.backend.dot(inputs, self.wa) + self.ba
        wgts = tf.math.sin(dp)
        ret = tf.concat([bias, wgts], -1)
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.k + 1)

# TODO: Load your data here. Replace with actual data loading code.
# For example: df = pd.read_csv('your_data.csv')
# Assume df is a DataFrame with columns like 'Reference Key', 'LIS Result: Numeric Result', 'LIS Reference Datetime', 'label'
df = pd.DataFrame()  # Placeholder - replace with actual data

group = df.groupby('Reference Key')

# Check labels are consistent per patient (optional, for debugging)
for key, grp in group:
    if len(grp['label'].unique()) > 1:
        print(f'Inconsistent label for patient {key}')

# Patients for new df capping high PSA value to 10
new_df = df.copy()
new_df['LIS Result: Numeric Result'] = np.minimum(new_df['LIS Result: Numeric Result'], 10)
group_new = new_df.groupby('Reference Key')

df_versions = ['original', 'filtered']
results = []

for df_version in df_versions:
    if df_version == 'original':
        cur_group = group
        cur_patients = list(group.groups.keys())
    else:
        cur_group = group_new
        cur_patients = list(group_new.groups.keys())

    for n in Ns:
        for cutoff in cutoffs:
            print(f"\n=== Processing {df_version} df, n={n}, cutoff={cutoff} ===")
            
            X_list = []
            y_list = []
            patient_list = []
            label_list = []
            for key in cur_patients:
                grp = cur_group.get_group(key).sort_values('LIS Reference Datetime')
                if len(grp) < n + 1:
                    continue
                psa = grp['LIS Result: Numeric Result'].values
                times_str = grp['LIS Reference Datetime'].values
                try:
                    times = np.array([pd.to_datetime(t) for t in times_str])
                except Exception as e:
                    print(f"Date parsing error for patient {key}: {e}")
                    continue
                label = grp['label'].iloc[0]
                if np.all(psa[:n] <= cutoff):
                    # Compute deltas and velocities
                    deltas = [0.0]
                    velocities = [0.0]
                    for j in range(1, n):
                        delta = (times[j] - times[j-1]) / np.timedelta64(1, 'D')
                        deltas.append(delta)
                        vel = (psa[j] - psa[j-1]) / delta if delta > 0 else 0.0
                        velocities.append(vel)
                    # Features: (n, 3) - psa, velocity, delta
                    features = np.stack([psa[:n], velocities, deltas], axis=1)
                    X_list.append(features)
                    y = 1 if psa[n] > cutoff else 0
                    y_list.append(y)
                    patient_list.append(key)
                    label_list.append(label)

            if len(X_list) == 0:
                print("No data for this combination. Skipping.")
                continue

            X = np.array(X_list)
            y = np.array(y_list)
            num_patients_N_below = len(X_list)
            cancer_in_N_below = sum(label_list)
            indices_cross = [i for i in range(len(y_list)) if y_list[i] == 1]
            num_patients_cross = len(indices_cross)
            cancer_in_cross = sum(label_list[i] for i in indices_cross)
            print(f"Patients with first {n} <= {cutoff}: {num_patients_N_below} (cancer: {cancer_in_N_below})")
            print(f"Patients crossing cutoff at {n+1}: {num_patients_cross} (cancer: {cancer_in_cross})")

            if len(np.unique(y)) < 2:
                print("Only one class present. Skipping.")
                continue

            # Balance the dataset by downsampling the majority class (negatives) to match positives
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            if len(pos_idx) > 0 and len(neg_idx) > len(pos_idx):
                balanced_neg_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
                balanced_idx = np.concatenate([pos_idx, balanced_neg_idx])
                np.random.shuffle(balanced_idx)  # Shuffle for randomness in CV
                X_bal = X[balanced_idx]
                y_bal = y[balanced_idx]
                patient_bal = np.array(patient_list)[balanced_idx]
                label_bal = np.array(label_list)[balanced_idx]
                balanced_num_patients = len(X_bal)
                balanced_num_cross = sum(y_bal)
                print(f"Balanced dataset size: {balanced_num_patients} (crossings: {balanced_num_cross})")
            else:
                X_bal = X
                y_bal = y
                patient_bal = np.array(patient_list)
                label_bal = np.array(label_list)
                balanced_num_patients = num_patients_N_below
                balanced_num_cross = num_patients_cross
                print("No balancing needed or possible.")

            if len(np.unique(y_bal)) < 2:
                print("Only one class present after balancing. Skipping.")
                continue

            # K-fold cross validation on balanced data
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
            fold_acc = []
            fold_prec = []
            fold_rec = []
            fold_f1 = []
            fold_auc = []

            for fold, (train_full_idx, test_idx) in enumerate(skf.split(X_bal, y_bal)):
                print(f"\n--- Fold {fold+1}/{k_folds} ---")
                X_train_full, X_test = X_bal[train_full_idx], X_bal[test_idx]
                y_train_full, y_test = y_bal[train_full_idx], y_bal[test_idx]
                patients_train_full, patients_test = patient_bal[train_full_idx], patient_bal[test_idx]

                if len(X_train_full) == 0 or len(X_test) == 0:
                    print("Empty train or test set in fold. Skipping fold.")
                    continue

                # Split train_full into train and val
                train_idx, val_idx = train_test_split(range(len(y_train_full)), test_size=validation_split, stratify=y_train_full, random_state=random_state)
                X_train = X_train_full[train_idx]
                X_val = X_train_full[val_idx]
                y_train = y_train_full[train_idx]
                y_val = y_train_full[val_idx]
                patients_train = patients_train_full[train_idx]
                patients_val = patients_train_full[val_idx]

                print(f"Training set: {X_train.shape[0]} samples (positives: {sum(y_train)})")
                print(f"Unique reference keys in training: {len(set(patients_train))}")
                print(f"Number of rows in training data: {len(patients_train) * (n + 1)}")
                print(f"Validation set: {X_val.shape[0]} samples (positives: {sum(y_val)})")
                print(f"Unique reference keys in validation: {len(set(patients_val))}")
                print(f"Number of rows in validation data: {len(patients_val) * (n + 1)}")
                print(f"Test set: {X_test.shape[0]} samples (positives: {sum(y_test)})")
                print("Starting model training...")

                # Compute class weights on train
                classes = np.unique(y_train)
                class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
                class_weight_dict = dict(zip(classes, class_weights))

                # Build model
                def build_model(n_timesteps, feat_dim=3, t2v_k=t2v_kernel_size, lstm_units=lstm_units):
                    input_layer = Input(shape=(n_timesteps, feat_dim))
                    time_feat = input_layer[:, :, 2:3]
                    t2v = Time2Vec(kernel_size=t2v_k)(time_feat)
                    other_feats = input_layer[:, :, :2]
                    embedded = Concatenate()([other_feats, t2v])
                    lstm_out = LSTM(lstm_units)(embedded)
                    output = Dense(1, activation='sigmoid')(lstm_out)
                    model = Model(inputs=input_layer, outputs=output)
                    optimizer = Adam(learning_rate=0.001)
                    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                    return model

                model = build_model(n)
                early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stop], class_weight=class_weight_dict)

                # Tune threshold on val
                y_val_pred_prob = model.predict(X_val).flatten()
                precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred_prob)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5

                # Evaluate on test with tuned threshold
                y_pred_prob = model.predict(X_test).flatten()
                y_pred = (y_pred_prob > optimal_threshold).astype(int)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test)) > 1 else 0.0
                cm = confusion_matrix(y_test, y_pred)

                fold_acc.append(acc)
                fold_prec.append(prec)
                fold_rec.append(rec)
                fold_f1.append(f1)
                fold_auc.append(auc)

                print(f"\nFold Evaluation Results (with auto-tuned threshold {round(optimal_threshold, 3)}):")
                print(f"Accuracy: {round(acc, 3)}")
                print(f"Precision: {round(prec, 3)}")
                print(f"Recall: {round(rec, 3)}")
                print(f"F1-Score: {round(f1, 3)}")
                print(f"AUC-ROC: {round(auc, 3)}")
                print("Confusion Matrix:")
                print(cm)

                # Save train validation loss plot
                plt.figure(figsize=(8, 6))
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title(f'Loss Curve - {df_version}, n={n}, cutoff={cutoff}, Fold {fold+1}')
                plt.savefig(f'results_folder/loss_plot_{df_version}_n{n}_cutoff{cutoff}_fold{fold+1}.png')
                plt.close()

                # Save ROC curve plot (includes AUC-ROC)
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {df_version}, n={n}, cutoff={cutoff}, Fold {fold+1}')
                plt.legend(loc="lower right")
                plt.savefig(f'results_folder/roc_plot_{df_version}_n{n}_cutoff{cutoff}_fold{fold+1}.png')
                plt.close()

                # Save confusion matrix plot
                plt.figure(figsize=(5, 4))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix - {df_version}, n={n}, cutoff={cutoff}, Fold {fold+1}')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['No Cross', 'Cross'], rotation=45)
                plt.yticks(tick_marks, ['No Cross', 'Cross'])
                thresh = cm.max() / 2.
                for i, j in np.ndindex(cm.shape):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig(f'results_folder/cm_plot_{df_version}_n{n}_cutoff{cutoff}_fold{fold+1}.png')
                plt.close()

            if len(fold_acc) == 0:
                print("No valid folds. Skipping.")
                continue

            # Average metrics
            avg_acc = round(np.mean(fold_acc), 3)
            avg_prec = round(np.mean(fold_prec), 3)
            avg_rec = round(np.mean(fold_rec), 3)
            avg_f1 = round(np.mean(fold_f1), 3)
            avg_auc = round(np.mean(fold_auc), 3)

            print("\nAverage Evaluation Results across folds:")
            print(f"Accuracy: {avg_acc}")
            print(f"Precision: {avg_prec}")
            print(f"Recall: {avg_rec}")
            print(f"F1-Score: {avg_f1}")
            print(f"AUC-ROC: {avg_auc}")

            results.append({
                'df_used': df_version,
                'n': n,
                'cutoff': cutoff,
                'num_patients_N_below': num_patients_N_below,
                'cancer_in_N_below': cancer_in_N_below,
                'num_patients_cross': num_patients_cross,
                'cancer_in_cross': cancer_in_cross,
                'balanced_num_patients': balanced_num_patients,
                'balanced_num_cross': balanced_num_cross,
                'acc': avg_acc,
                'prec': avg_prec,
                'rec': avg_rec,
                'f1': avg_f1,
                'auc': avg_auc
            })

results_df = pd.DataFrame(results)
results_df.to_csv('results_folder/results_df.csv', index=False)
print("\nAll results saved to 'results_folder/results_df.csv'")