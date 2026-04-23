from __future__ import print_function

import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

plt.ion()

batch_size = 12
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True

train_dir = r'C:\Users\adeif\OneDrive\Year 4\Sem 2\Computer Vision\Assignment 2\chest_xray\train'
test_dir = r'C:\Users\adeif\OneDrive\Year 4\Sem 2\Computer Vision\Assignment 2\chest_xray\test'


def get_gradcam_heatmap(feature_extractor, classifier, image_array, pred_index=None):
    """
    Generates a Grad-CAM heatmap for one image.
    """
    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(image_array)
        tape.watch(conv_outputs)

        predictions = classifier(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


with tf.device('/cpu:0'):

    # Datasets
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("Class Names:", class_names)

    # Class weights
    train_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )

    class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
    print("Class weights:", class_weights)

    # Sample training images
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # Main model
    model = tf.keras.models.Sequential([
        data_augmentation,
        Rescaling(1.0 / 255),
        Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu', name='last_conv_layer'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    save_callback = tf.keras.callbacks.ModelCheckpoint(
        "pneumonia.keras",
        save_best_only=True
    )

    if fit:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            callbacks=[save_callback, earlystop_callback],
            epochs=epochs,
            class_weight=class_weights
        )
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    # Evaluation
    score = model.evaluate(test_ds, batch_size=batch_size)
    print("Test accuracy:", score[1])

    # Accuracy graph
    if fit:
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plt.savefig("accuracy_graph.png")
        plt.show()
        plt.pause(0.1)

    # Metrics
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    plt.pause(0.1)

    # Sample predictions
    plt.figure(figsize=(10, 10))
    for images, labels in test_ds.take(1):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(), 0), verbose=0)
            plt.title(
                "Actual: " + class_names[labels[i].numpy()] +
                "\nPredicted: {} {:.2f}%".format(
                    class_names[np.argmax(prediction)],
                    100 * np.max(prediction)
                )
            )
            plt.axis("off")
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.show()
    plt.pause(0.1)

    # ----------------------------
    # Grad-CAM
    # ----------------------------
    print("\n--- GRAD-CAM START ---")

    # Build feature extractor and classifier separately
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("last_conv_layer").output
    )

    classifier_input = tf.keras.Input(shape=feature_extractor.output.shape[1:])
    x = classifier_input
    start_collecting = False

    for layer in model.layers:
        if layer.name == "last_conv_layer":
            start_collecting = True
            continue
        if start_collecting:
            x = layer(x)

    classifier = tf.keras.Model(classifier_input, x)

    for images, labels in test_ds.take(1):
        image = images[0].numpy()
        image_array = tf.expand_dims(images[0], axis=0)

        preds = model.predict(image_array, verbose=0)
        pred_class = np.argmax(preds[0])
        actual_class = labels[0].numpy()

        print("Grad-CAM Actual:", class_names[actual_class])
        print("Grad-CAM Predicted:", class_names[pred_class])

        heatmap = get_gradcam_heatmap(feature_extractor, classifier, image_array, pred_class)
        print("Heatmap generated successfully. Shape:", heatmap.shape)

        # Original
        plt.figure(figsize=(5, 5))
        plt.imshow(image.astype("uint8"))
        plt.title(f"Original Image - Actual: {class_names[actual_class]}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("gradcam_original.png")
        plt.show()
        plt.pause(0.1)

        # Heatmap
        plt.figure(figsize=(5, 5))
        plt.imshow(heatmap, cmap="jet")
        plt.title("Grad-CAM Heatmap")
        plt.colorbar()
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("gradcam_heatmap.png")
        plt.show()
        plt.pause(0.1)

        # Overlay
        plt.figure(figsize=(5, 5))
        plt.imshow(image.astype("uint8"))
        plt.imshow(
            heatmap,
            cmap="jet",
            alpha=0.4,
            extent=(0, image.shape[1], image.shape[0], 0)
        )
        plt.title("Grad-CAM Overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("gradcam_overlay.png")
        plt.show()
        plt.pause(0.1)

        break

