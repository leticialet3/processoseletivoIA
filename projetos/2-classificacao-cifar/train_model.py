import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    tf.keras.utils.set_random_seed(42)

    # 1. Carregamento e Normalização
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 2. Validação
    x_val, y_val = x_train[-5000:], y_train[-5000:]
    x_train, y_train = x_train[:-5000], y_train[:-5000]

    # 3. Construção do Modelo 
    model = keras.Sequential([
        # Data Augmentation embutida no modelo
        layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # Bloco 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloco 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloco 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Cabeçote do Modelo
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Callbacks 
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=5, 
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True)
    ]

    # 5. Treinamento
    print("Iniciando treinamento...")
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=25,
        batch_size=64,
        callbacks=callbacks
    )

    # 6. Exibição da Acurácia de Validação Final
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"\n---> Acurácia Final de Validação: {val_acc * 100:.2f}% <---")

if __name__ == "__main__":
    main()