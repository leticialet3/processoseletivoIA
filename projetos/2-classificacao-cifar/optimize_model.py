import tensorflow as tf

def main():
    print("Carregando o arquivo 'model.h5'...")
    model = tf.keras.models.load_model("model.h5")

    # Configuração do Conversor TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Aplicação da Otimização (Dynamic Range Quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Otimizando e convertendo para TensorFlow Lite...")
    tflite_model = converter.convert()

    # Salvando arquivo .tflite
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    print("Sucesso! O modelo otimizado foi salvo como 'model.tflite'.")

if __name__ == "__main__":
    main()