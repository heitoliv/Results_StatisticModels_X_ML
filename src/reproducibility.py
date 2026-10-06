import os
import random

import numpy as np
import tensorflow as tf


def configure_tensorflow_gpu() -> list[str]:
    """Habilita GPUs TensorFlow sem reservar toda a memória de vídeo.

    A configuração deve ser chamada antes da criação/treinamento de qualquer
    modelo Keras. Retorna os dispositivos lógicos que serão usados.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPU TensorFlow não encontrada; a LSTM será executada em CPU.")
        return []

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        # A configuração só pode ser modificada antes de inicializar a GPU.
        print(f"GPU detectada, mas a configuração de memória já foi inicializada: {error}")

    logical_gpus = tf.config.list_logical_devices("GPU")
    devices = [device.name for device in logical_gpus]
    print(f"GPU TensorFlow habilitada: {', '.join(devices)}")
    return devices


def set_seed(seed: int) -> None:
    """Define as seeds usadas nos componentes principais do experimento."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
