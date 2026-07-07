import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

# ==========================================
# 1. DATA LOADING & VALIDATION
# ==========================================
print("--- Step 1: Loading and Validating Data ---")
dataset_path = r"C:\Users\barry\Documents\fyp2 dental\xray image"
train_path, valid_path, test_path = [os.path.join(dataset_path, x) for x in ["train", "valid", "test"]]

train_df = pd.read_csv(os.path.join(train_path, "_annotations.csv"))
valid_df = pd.read_csv(os.path.join(valid_path, "_annotations.csv"))
test_df = pd.read_csv(os.path.join(test_path, "_annotations.csv"))

# Clean invalid bounding boxes
for df in [train_df, valid_df, test_df]:
    df.drop(df[(df["xmin"] < 0) | (df["ymin"] < 0) | (df["xmax"] <= df["xmin"]) | (df["ymax"] <= df["ymin"])].index, inplace=True)

# Binary Labels
target_class = "Cavity"
train_cavity_imgs = set(train_df[train_df["class"] == target_class]["filename"].unique())
valid_cavity_imgs = set(valid_df[valid_df["class"] == target_class]["filename"].unique())
test_cavity_imgs = set(test_df[test_df["class"] == target_class]["filename"].unique())

def get_files_labels(folder, cavity_set):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    labels = [1 if f in cavity_set else 0 for f in files]
    return files, np.array(labels)

train_files, y_train = get_files_labels(train_path, train_cavity_imgs)
valid_files, y_valid = get_files_labels(valid_path, valid_cavity_imgs)
test_files, y_test = get_files_labels(test_path, test_cavity_imgs)

# Preprocessing Function (Grayscale -> CLAHE -> Resize -> Normalize)
def load_and_preprocess(files, folder, img_size=(224, 224)):
    X = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for f in files:
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = clahe.apply(img) / 255.0
            X.append(img)
    return np.array(X, dtype=np.float32)

X_train = np.expand_dims(load_and_preprocess(train_files, train_path), axis=-1)
X_valid = np.expand_dims(load_and_preprocess(valid_files, valid_path), axis=-1)
X_test = np.expand_dims(load_and_preprocess(test_files, test_path), axis=-1)

class_weights = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)))

# ==========================================
# 2. BASELINE CNN (WITH GAP & AUGMENTATION)
# ==========================================
print("--- Step 2: Training Baseline CNN ---")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

baseline_model = Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 1)),
    data_augmentation, 
    Conv2D(32, (3, 3), activation="relu"), BatchNormalization(), MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"), BatchNormalization(), MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"), BatchNormalization(), MaxPooling2D((2, 2)),
    GlobalAveragePooling2D(), 
    Dense(128, activation="relu"), Dropout(0.5),
    Dense(1, activation="sigmoid")
])

baseline_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
baseline_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=20, batch_size=16, 
                   class_weight=class_weights, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=1)

# ==========================================
# 3. MOBILENETV2 (WITH FINE-TUNING)
# ==========================================
print("--- Step 3: Training MobileNetV2 ---")
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_valid_rgb = np.repeat(X_valid, 3, axis=-1)
X_test_rgb = np.repeat(X_test, 3, axis=-1)

X_train_mob = mobilenet_preprocess(X_train_rgb * 255.0)
X_valid_mob = mobilenet_preprocess(X_valid_rgb * 255.0)
X_test_mob = mobilenet_preprocess(X_test_rgb * 255.0)

base_mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_mobilenet.trainable = False 

x = GlobalAveragePooling2D()(base_mobilenet.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
mob_output = Dense(1, activation="sigmoid")(x)
mob_model = Model(inputs=base_mobilenet.input, outputs=mob_output)

mob_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
mob_model.fit(X_train_mob, y_train, validation_data=(X_valid_mob, y_valid), epochs=10, batch_size=16, 
              class_weight=class_weights, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)

# Fine-tuning Stage 2
base_mobilenet.trainable = True
for layer in base_mobilenet.layers[:-20]:
    layer.trainable = False
mob_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
mob_model.fit(X_train_mob, y_train, validation_data=(X_valid_mob, y_valid), epochs=10, batch_size=16, 
              class_weight=class_weights, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)

# ==========================================
# 4. RESNET50 (WITH FINE-TUNING)
# ==========================================
print("--- Step 4: Training ResNet50 ---")
X_train_res = resnet_preprocess(X_train_rgb * 255.0)
X_valid_res = resnet_preprocess(X_valid_rgb * 255.0)
X_test_res = resnet_preprocess(X_test_rgb * 255.0)

base_resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_resnet.trainable = False

x = GlobalAveragePooling2D()(base_resnet.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
res_output = Dense(1, activation="sigmoid")(x)
res_model = Model(inputs=base_resnet.input, outputs=res_output)

res_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
res_model.fit(X_train_res, y_train, validation_data=(X_valid_res, y_valid), epochs=10, batch_size=16, 
              class_weight=class_weights, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)

# Fine-tuning Stage 2
base_resnet.trainable = True
for layer in base_resnet.layers[:-20]:
    layer.trainable = False
res_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
res_model.fit(X_train_res, y_train, validation_data=(X_valid_res, y_valid), epochs=10, batch_size=16, 
              class_weight=class_weights, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)

# ==========================================
# 5. EVALUATION & MODEL SELECTION (F1-SCORE)
# ==========================================
print("--- Step 5: Evaluating Models & Selecting Best based on F1-Score ---")
models_to_test = {
    "Baseline CNN (GAP)": (X_test, baseline_model),
    "MobileNetV2": (X_test_mob, mob_model),
    "ResNet50": (X_test_res, res_model)
}

best_f1 = -1
best_model_name = ""
best_model_obj = None
best_X_test = None

results = []
for name, (X_t, model) in models_to_test.items():
    y_prob = model.predict(X_t).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=["No Cavity", "Cavity"], zero_division=0)
    
    cavity_f1 = report['Cavity']['f1-score']
    
    results.append({
        "Model": name,
        "Accuracy": report['accuracy'],
        "Cavity Precision": report['Cavity']['precision'],
        "Cavity Recall": report['Cavity']['recall'],
        "Cavity F1-Score": cavity_f1
    })
    
    if cavity_f1 > best_f1:
        best_f1 = cavity_f1
        best_model_name = name
        best_model_obj = model
        best_X_test = X_t

df_results = pd.DataFrame(results)
print("\n--- FINAL MODEL COMPARISON ---")
print(df_results)
print(f"\n✅ WINNER: {best_model_name} (F1-Score: {best_f1:.3f})")

best_model_obj.save("best_caries_model.keras")
print(f"Saved '{best_model_name}' as 'best_caries_model.keras'")

# ==========================================
# 6. ROC CURVE & THRESHOLD TUNING
# ==========================================
y_prob_best = best_model_obj.predict(best_X_test).flatten()
fpr, tpr, thresholds = roc_curve(y_test, y_prob_best)
roc_auc = roc_auc_score(y_test, y_prob_best)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal Medical Threshold: {optimal_threshold:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve'); plt.legend(); plt.savefig("roc_curve.png"); plt.show()

# ==========================================
# 7. DYNAMIC GRAD-CAM (FIXED DUAL-OUTPUT)
# ==========================================
print("--- Step 7: Generating Grad-CAM ---")

def generate_gradcam(model, image_array):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
            
    if last_conv_layer is None:
        return None

    # Dual output model ensures gradients can flow from prediction to conv layer
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(image_array)
        loss = preds[:, 0]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

cavity_indices = np.where(y_test == 1)[0]
if len(cavity_indices) > 0:
    sample_img = best_X_test[cavity_indices[0]]
    sample_img_batch = np.expand_dims(sample_img, axis=0)
    
    heatmap = generate_gradcam(best_model_obj, sample_img_batch)
    if heatmap is not None:
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        original_rgb = cv2.cvtColor((sample_img.squeeze() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        superimposed = cv2.addWeighted(original_rgb, 0.6, heatmap_colored, 0.4, 0)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.title("Original"); plt.imshow(original_rgb); plt.axis('off')
        plt.subplot(1, 2, 2); plt.title("Grad-CAM"); plt.imshow(superimposed); plt.axis('off')
        plt.tight_layout(); plt.savefig("gradcam_result.png"); plt.show()

print("\n✅ MASTER PIPELINE COMPLETE!")
