from __future__ import print_function

import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


batch_size = 12
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True  # make fit false if you do not want to train the network again

train_dir = r'C:\Users\adeif\OneDrive\Year 4\Sem 2\Computer Vision\Assignment 2\chest_xray\train'
test_dir = r'C:\Users\adeif\OneDrive\Year 4\Sem 2\Computer Vision\Assignment 2\chest_xray\test'


def get_gradcam_heatmap(model, image, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()


with tf.device('/cpu:0'):

    # create training, validation and test datasets
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

    print('Class Names:', class_names)

    # compute class weights
    train_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )

    class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
    print("Class weights:", class_weights)

    # show sample training images
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()

    # data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # create model
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

    # evaluate on test set
    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])

    # plot training history
    if fit:
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    # predictions for metrics
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
    plt.show()

    # show sample predictions
    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(), 0), verbose=0)
            plt.title(
                'Actual: ' + class_names[labels[i].numpy()] +
                '\nPredicted: {} {:.2f}%'.format(
                    class_names[np.argmax(prediction)],
                    100 * np.max(prediction)
                )
            )
            plt.axis("off")
    plt.show()

    # Grad-CAM example
    for images, labels in test_ds.take(1):
        image = images[0]
        label = labels[0].numpy()
        image_array = tf.expand_dims(image, axis=0)

        heatmap = get_gradcam_heatmap(model, image_array, "last_conv_layer")

        plt.figure(figsize=(5, 5))
        plt.imshow(image.numpy().astype("uint8"))
        plt.title(f"Original Image - Actual: {class_names[label]}")
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(heatmap, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.colorbar()
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.imshow(image.numpy().astype("uint8"))
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.title("Grad-CAM Overlay")
        plt.axis("off")
        plt.show()

        break

        from tensorflow.keras.applications import MobileNetV2

        base_model = MobileNetV2(
            input_shape=(img_height, img_width, img_channels),
            include_top=False,
            weights='imagenet'
        )

        # freeze base model
        base_model.trainable = False

        model = tf.keras.Sequential([
            data_augmentation,
            Rescaling(1.0 / 255),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])