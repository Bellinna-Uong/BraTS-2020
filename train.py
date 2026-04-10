
"""
Commented training script for 3D brain tumor segmentation on BraTS-style data.

This version is designed for reporting and experimentation:
- detailed comments in English
- saved plots for the report
- saved training history and evaluation scores
- saved sample predictions
- optional class-weight computation
"""

import os
import json
import glob
import random
import csv
import numpy as np
import pandas as pd
import keras
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.metrics import MeanIoU

from custom_datagen import imageLoader
from simple_3d_unet_commented import simple_unet_model
import segmentation_models_3D as sm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "training_outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

for folder in [OUTPUT_DIR, PLOTS_DIR, PRED_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(folder, exist_ok=True)

train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"
val_img_dir = "BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_dir = "BraTS2020_TrainingData/input_data_128/val/masks/"

train_img_list = sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))
val_img_list = sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))

print("Train images:", len(train_img_list))
print("Train masks :", len(train_mask_list))
print("Val images  :", len(val_img_list))
print("Val masks   :", len(val_mask_list))

def save_sample_modalities_and_mask(image, mask_one_hot, out_path, title_prefix="sample"):
    if mask_one_hot.ndim == 4:
        mask = np.argmax(mask_one_hot, axis=3)
    else:
        mask = mask_one_hot

    n_slice = random.randint(0, mask.shape[2] - 1)

    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(image[:, :, n_slice, 0], cmap="gray")
    plt.title(f"{title_prefix} - FLAIR")

    plt.subplot(222)
    plt.imshow(image[:, :, n_slice, 1], cmap="gray")
    plt.title(f"{title_prefix} - T1ce")

    plt.subplot(223)
    plt.imshow(image[:, :, n_slice, 2], cmap="gray")
    plt.title(f"{title_prefix} - T2")

    plt.subplot(224)
    plt.imshow(mask[:, :, n_slice])
    plt.title(f"{title_prefix} - Mask")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_prediction_comparison(image, true_mask, pred_mask, out_path, n_slice=55):
    max_slice = image.shape[2] - 1
    n_slice = min(n_slice, max_slice)

    plt.figure(figsize=(12, 8))

    plt.subplot(131)
    plt.title("Testing Image (T1ce)")
    plt.imshow(image[:, :, n_slice, 1], cmap="gray")

    plt.subplot(132)
    plt.title("Ground Truth")
    plt.imshow(true_mask[:, :, n_slice])

    plt.subplot(133)
    plt.title("Prediction")
    plt.imshow(pred_mask[:, :, n_slice])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_history_csv(history, out_csv_path):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(out_csv_path, index=False)
    return history_df

def save_training_curves(history, out_prefix):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist["loss"], label="Training loss")
    if "val_loss" in hist:
        plt.plot(epochs, hist["val_loss"], label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_loss.png", bbox_inches="tight")
    plt.close()

    if "accuracy" in hist:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, hist["accuracy"], label="Training accuracy")
        if "val_accuracy" in hist:
            plt.plot(epochs, hist["val_accuracy"], label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_accuracy.png", bbox_inches="tight")
        plt.close()

    iou_train_key = None
    iou_val_key = None
    for key in hist.keys():
        lower_key = key.lower()
        if "iou" in lower_key and not lower_key.startswith("val_"):
            iou_train_key = key
        if "iou" in lower_key and lower_key.startswith("val_"):
            iou_val_key = key

    if iou_train_key is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, hist[iou_train_key], label=f"Training {iou_train_key}")
        if iou_val_key is not None:
            plt.plot(epochs, hist[iou_val_key], label=f"Validation {iou_val_key}")
        plt.title("Training and validation IoU")
        plt.xlabel("Epochs")
        plt.ylabel("IoU")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_iou.png", bbox_inches="tight")
        plt.close()

def save_scores_json(scores_dict, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f, indent=4)

def save_scores_csv(scores_dict, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in scores_dict.items():
            writer.writerow([key, value])

num_images = len(train_img_list)
img_num = random.randint(0, num_images - 1)

test_img = np.load(os.path.join(train_img_dir, train_img_list[img_num]))
test_mask = np.load(os.path.join(train_mask_dir, train_mask_list[img_num]))

print("Random sample image shape:", test_img.shape)
print("Random sample mask shape :", test_mask.shape)

save_sample_modalities_and_mask(
    test_img,
    test_mask,
    os.path.join(PLOTS_DIR, "random_training_sample.png"),
    title_prefix="Random training sample"
)

def compute_class_weights(mask_dir):
    columns = ["0", "1", "2", "3"]
    df = pd.DataFrame(columns=columns)

    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))

    for idx, mask_path in enumerate(mask_paths):
        print(f"Reading mask {idx + 1}/{len(mask_paths)}")
        temp_mask = np.load(mask_path)
        temp_mask = np.argmax(temp_mask, axis=3)

        values, counts = np.unique(temp_mask, return_counts=True)
        count_dict = {str(v): c for v, c in zip(values, counts)}

        for col in columns:
            if col not in count_dict:
                count_dict[col] = 0

        df.loc[len(df)] = count_dict

    label_0 = df["0"].sum()
    label_1 = df["1"].sum()
    label_2 = df["2"].sum()
    label_3 = df["3"].sum()

    total_labels = label_0 + label_1 + label_2 + label_3
    n_classes = 4

    wt0 = round((total_labels / (n_classes * label_0)), 2)
    wt1 = round((total_labels / (n_classes * label_1)), 2)
    wt2 = round((total_labels / (n_classes * label_2)), 2)
    wt3 = round((total_labels / (n_classes * label_3)), 2)

    counts_dict = {
        "label_0": int(label_0),
        "label_1": int(label_1),
        "label_2": int(label_2),
        "label_3": int(label_3),
        "weight_0": float(wt0),
        "weight_1": float(wt1),
        "weight_2": float(wt2),
        "weight_3": float(wt3),
    }

    return counts_dict

class_stats = compute_class_weights(train_mask_dir)
print("Class statistics:", class_stats)

save_scores_json(class_stats, os.path.join(REPORTS_DIR, "class_distribution_and_weights.json"))
save_scores_csv(class_stats, os.path.join(REPORTS_DIR, "class_distribution_and_weights.csv"))

plt.figure(figsize=(8, 5))
plt.bar(
    ["Class 0", "Class 1", "Class 2", "Class 3"],
    [
        class_stats["label_0"],
        class_stats["label_1"],
        class_stats["label_2"],
        class_stats["label_3"],
    ]
)
plt.title("Voxel count per class in training masks")
plt.xlabel("Classes")
plt.ylabel("Voxel count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), bbox_inches="tight")
plt.close()

batch_size = 2

train_img_datagen = imageLoader(
    train_img_dir, train_img_list,
    train_mask_dir, train_mask_list,
    batch_size
)

val_img_datagen = imageLoader(
    val_img_dir, val_img_list,
    val_mask_dir, val_mask_list,
    batch_size
)

img_batch, mask_batch = train_img_datagen.__next__()
print("Image batch shape:", img_batch.shape)
print("Mask batch shape :", mask_batch.shape)

batch_img_num = random.randint(0, img_batch.shape[0] - 1)
save_sample_modalities_and_mask(
    img_batch[batch_img_num],
    mask_batch[batch_img_num],
    os.path.join(PLOTS_DIR, "generator_batch_sample.png"),
    title_prefix="Generator batch sample"
)

wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [
    "accuracy",
    sm.metrics.IOUScore(threshold=0.5)
]

learning_rate = 0.0001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

model = simple_unet_model(
    IMG_HEIGHT=128,
    IMG_WIDTH=128,
    IMG_DEPTH=128,
    IMG_CHANNELS=3,
    num_classes=4
)

model.compile(optimizer=optimizer, loss=total_loss, metrics=metrics)

model_summary_path = os.path.join(REPORTS_DIR, "model_summary.txt")
with open(model_summary_path, "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print(model.summary())
print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size
epochs = 100

history = model.fit(
    train_img_datagen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    validation_data=val_img_datagen,
    validation_steps=val_steps_per_epoch,
)

trained_model_path = os.path.join(MODELS_DIR, "brats_3d_trained.hdf5")
model.save(trained_model_path)
print("Model saved to:", trained_model_path)

history_csv_path = os.path.join(REPORTS_DIR, "training_history.csv")
history_df = save_history_csv(history, history_csv_path)
print("Training history saved to:", history_csv_path)

save_training_curves(history, os.path.join(PLOTS_DIR, "training"))
print("Training curves saved in:", PLOTS_DIR)

final_scores = {
    "random_seed": RANDOM_SEED,
    "batch_size": batch_size,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "steps_per_epoch": steps_per_epoch,
    "val_steps_per_epoch": val_steps_per_epoch,
    "final_train_loss": float(history.history["loss"][-1]),
    "final_val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
    "final_train_accuracy": float(history.history["accuracy"][-1]) if "accuracy" in history.history else None,
    "final_val_accuracy": float(history.history["val_accuracy"][-1]) if "val_accuracy" in history.history else None,
}

for key, value in history.history.items():
    if "iou" in key.lower():
        final_scores[f"final_{key}"] = float(value[-1])

save_scores_json(final_scores, os.path.join(REPORTS_DIR, "final_training_scores.json"))
save_scores_csv(final_scores, os.path.join(REPORTS_DIR, "final_training_scores.csv"))

inference_model = load_model(trained_model_path, compile=False)

eval_batch_size = 8

test_img_datagen = imageLoader(
    val_img_dir, val_img_list,
    val_mask_dir, val_mask_list,
    eval_batch_size
)

test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = inference_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

mean_iou_metric = MeanIoU(num_classes=4)
mean_iou_metric.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
batch_mean_iou = float(mean_iou_metric.result().numpy())

evaluation_scores = {
    "validation_batch_size": eval_batch_size,
    "batch_mean_iou": batch_mean_iou
}

print("Mean IoU on validation batch =", batch_mean_iou)

save_scores_json(evaluation_scores, os.path.join(REPORTS_DIR, "evaluation_scores.json"))
save_scores_csv(evaluation_scores, os.path.join(REPORTS_DIR, "evaluation_scores.csv"))

example_index = 0

test_img = np.load(os.path.join(val_img_dir, val_img_list[example_index]))
test_mask = np.load(os.path.join(val_mask_dir, val_mask_list[example_index]))
test_mask_argmax = np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = inference_model.predict(test_img_input)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0]

save_prediction_comparison(
    test_img,
    test_mask_argmax,
    test_prediction_argmax,
    os.path.join(PRED_DIR, f"prediction_comparison_{example_index}.png"),
    n_slice=55
)

np.save(os.path.join(PRED_DIR, f"prediction_mask_{example_index}.npy"), test_prediction_argmax)
np.save(os.path.join(PRED_DIR, f"ground_truth_mask_{example_index}.npy"), test_mask_argmax)
np.save(os.path.join(PRED_DIR, f"input_image_{example_index}.npy"), test_img)

run_summary = {
    "model_path": trained_model_path,
    "history_csv": history_csv_path,
    "plots_dir": PLOTS_DIR,
    "predictions_dir": PRED_DIR,
    "reports_dir": REPORTS_DIR,
    "batch_mean_iou": batch_mean_iou
}

save_scores_json(run_summary, os.path.join(REPORTS_DIR, "run_summary.json"))

print("\nRun completed successfully.")
print("All outputs are available in:", OUTPUT_DIR)
