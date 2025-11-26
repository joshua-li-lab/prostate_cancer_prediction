import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Layer
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# Create results directory
os.makedirs('results_folder', exist_ok=True)

# Parameters for N and cutoff
Ns = [4,5,6,7,8,9]
cutoffs = [4,5,6,7,8,9]

# Machine learning hyperparameters
t2v_kernel_size = 3
lstm_units = 128 
epochs = 20 
batch_size = 16 
validation_split = 0.1
random_state = 42
loss = 'sparse_categorical_crossentropy'
k_folds = 5
early_stop_patience = 5
num_classes = 3 # never, 0-12m, 12+m

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
            y_list = [] # 0: never, 1: 0-12m, 2: 12+m
            patient_list = []
            label_list = [] # cancer label
         
            for key in cur_patients:
                grp = cur_group.get_group(key).sort_values('LIS Reference Datetime')
                if len(grp) < n+1:
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
                    # Compute deltas and velocities for first n
                    deltas = [0.0]
                    velocities = [0.0]
                    for j in range(1, n):
                        delta = (times[j] - times[j-1]) / np.timedelta64(1, 'D')
                        deltas.append(delta)
                        vel = (psa[j] - psa[j-1]) / delta if delta > 0 else 0.0
                        velocities.append(vel)
                    # Features: (n, 3) - psa, velocity, delta
                    features = np.stack([psa[:n], velocities, deltas], axis=1)
                 
                    # Find first exceedance after n
                    exceed_idx = None
                    for k in range(n, len(psa)):
                        if psa[k] > cutoff:
                            exceed_idx = k
                            break
                 
                    if exceed_idx is None:
                        y = 0 # never
                    else:
                        time_to_exceed = (times[exceed_idx] - times[n-1]) / np.timedelta64(1, 'D')
                        if time_to_exceed <= 365:
                            y = 1 # 0-12 months
                        else:
                            y = 2 # 12+ months
                 
                    X_list.append(features)
                    y_list.append(y)
                    patient_list.append(key)
                    label_list.append(label)
         
            if len(X_list) == 0:
                print("No data for this combination. Skipping.")
                continue
         
            X = np.array(X_list)
            y = np.array(y_list)
            num_patients_total = len(X_list)
            cancer_total = sum(label_list)
         
            # Count per class
            class_counts = [sum(y == c) for c in range(num_classes)]
            num_never = class_counts[0]
            num_0_12 = class_counts[1]
            num_12_plus = class_counts[2]
            cancer_never = sum([label_list[i] for i in range(len(y)) if y[i] == 0])
            cancer_0_12 = sum([label_list[i] for i in range(len(y)) if y[i] == 1])
            cancer_12_plus = sum([label_list[i] for i in range(len(y)) if y[i] == 2])
         
            print(f"Patients with first {n} <= {cutoff}: {num_patients_total} (cancer: {cancer_total})")
            print(f"Never exceed: {num_never} (cancer: {cancer_never})")
            print(f"0-12 months: {num_0_12} (cancer: {cancer_0_12})")
            print(f"12+ months: {num_12_plus} (cancer: {cancer_12_plus})")
         
            if len(np.unique(y)) < 2:
                print("Fewer than 2 classes present. Skipping.")
                continue
         
            # Downsample to balance classes
            min_count = min(class_counts)
            balanced_idx = []
            for c in range(num_classes):
                class_idx = np.where(y == c)[0]
                if len(class_idx) > 0:
                    sampled_idx = np.random.choice(class_idx, size=min(min_count, len(class_idx)), replace=False)
                    balanced_idx.extend(sampled_idx)
         
            np.random.shuffle(balanced_idx)
            X_bal = X[balanced_idx]
            y_bal = y[balanced_idx]
            patient_bal = np.array(patient_list)[balanced_idx]
            label_bal = np.array(label_list)[balanced_idx]
            balanced_num_patients = len(X_bal)
            balanced_class_counts = [sum(y_bal == c) for c in range(num_classes)]
            balanced_num_never = balanced_class_counts[0]
            balanced_num_0_12 = balanced_class_counts[1]
            balanced_num_12_plus = balanced_class_counts[2]
         
            print(f"Balanced dataset size: {balanced_num_patients} (never: {balanced_num_never}, 0-12: {balanced_num_0_12}, 12+: {balanced_num_12_plus})")
         
            if len(np.unique(y_bal)) < 2:
                print("Fewer than 2 classes after balancing. Skipping.")
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
             
                print(f"Training set: {X_train.shape[0]} samples")
                print(f"Unique reference keys in training: {len(set(patients_train))}")
                print(f"Number of rows in training data: {len(patients_train) * (n + 1)}")
                print(f"Validation set: {X_val.shape[0]} samples")
                print(f"Unique reference keys in validation: {len(set(patients_val))}")
                print(f"Number of rows in validation data: {len(patients_val) * (n + 1)}")
                print(f"Test set: {X_test.shape[0]} samples")
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
                    output = Dense(3, activation='softmax')(lstm_out) # Changed to 3 units with softmax
                    model = Model(inputs=input_layer, outputs=output)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                    return model
                  
                model = build_model(n)
                early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
             
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stop], class_weight=class_weight_dict)
             
                # Tune thresholds on val for best F1 (adapted for multi-class using one-vs-rest)
                optimal_thresholds = []
                for c in range(num_classes):
                    y_val_bin = (y_val == c).astype(int)
                    y_val_prob_c = model.predict(X_val)[:, c]
                    precisions, recalls, thresholds = precision_recall_curve(y_val_bin, y_val_prob_c)
                    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
                    optimal_thresholds.append(optimal_threshold)
             
                # Evaluate on test using tuned thresholds
                y_pred_prob = model.predict(X_test)
                y_pred = []
                for i in range(len(y_pred_prob)):
                    probs = y_pred_prob[i]
                    candidates = [c for c in range(num_classes) if probs[c] > optimal_thresholds[c]]
                    if len(candidates) == 0:
                        y_pred.append(np.argmax(probs))
                    else:
                        y_pred.append(candidates[np.argmax([probs[c] for c in candidates])])
                y_pred = np.array(y_pred)
             
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                if len(np.unique(y_test)) > 1:
                    y_test_bin = label_binarize(y_test, classes=range(num_classes))
                    auc = roc_auc_score(y_test_bin, y_pred_prob, average='macro', multi_class='ovr')
                else:
                    auc = 0.0
                cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
             
                fold_acc.append(acc)
                fold_prec.append(prec)
                fold_rec.append(rec)
                fold_f1.append(f1)
                fold_auc.append(auc)
             
                print("\nFold Evaluation Results:")
                print(f"Accuracy: {round(acc, 3)}")
                print(f"Precision (macro): {round(prec, 3)}")
                print(f"Recall (macro): {round(rec, 3)}")
                print(f"F1-Score (macro): {round(f1, 3)}")
                print(f"AUC-ROC (macro, ovr): {round(auc, 3)}")
                print("Confusion Matrix:")
                print(cm)
             
                # Save train validation loss plot
                plt.figure(figsize=(8,6))
                plt.plot(history.history['loss'], label='Train Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title(f'Loss Curve - {df_version}, n={n}, cutoff={cutoff}, Fold {fold+1}')
                plt.savefig(f'results_folder/loss_plot_{df_version}_n{n}_cutoff{cutoff}_fold{fold+1}.png')
                plt.close()
             
                # Save confusion matrix plot
                plt.figure(figsize=(8,6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix - {df_version}, n={n}, cutoff={cutoff}, Fold {fold+1}')
                plt.colorbar()
                tick_marks = np.arange(num_classes)
                class_labels = ['Never', '0-12m', '12+m']
                plt.xticks(tick_marks, class_labels, rotation=45)
                plt.yticks(tick_marks, class_labels)
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
            print(f"Precision (macro): {avg_prec}")
            print(f"Recall (macro): {avg_rec}")
            print(f"F1-Score (macro): {avg_f1}")
            print(f"AUC-ROC (macro, ovr): {avg_auc}")
         
            results.append({
                'df_used': df_version,
                'n': n,
                'cutoff': cutoff,
                'num_patients_total': num_patients_total,
                'cancer_total': cancer_total,
                'num_never': num_never,
                'num_0_12': num_0_12,
                'num_12_plus': num_12_plus,
                'cancer_never': cancer_never,
                'cancer_0_12': cancer_0_12,
                'cancer_12_plus': cancer_12_plus,
                'balanced_num_patients': balanced_num_patients,
                'balanced_num_never': balanced_num_never,
                'balanced_num_0_12': balanced_num_0_12,
                'balanced_num_12_plus': balanced_num_12_plus,
                'acc': avg_acc,
                'prec': avg_prec,
                'rec': avg_rec,
                'f1': avg_f1,
                'auc': avg_auc
            })
results_df = pd.DataFrame(results)
results_df.to_csv('results_folder/results_df.csv', index=False)
print("\nAll results saved to 'results_folder/results_df.csv'")
