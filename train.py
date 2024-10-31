# Import yang diperlukan
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import numpy as np
import tensorflow as tf

# Callback untuk PSNR
class PSNRCallback(Callback):
    def __init__(self, valid_gen, model_path):
        super().__init__()
        self.valid_gen = valid_gen
        self.model_path = model_path
        self.best_psnr = -float('inf')  # Inisialisasi dengan nilai terburuk

    def on_epoch_end(self, epoch, logs=None):
        psnr_values = []
        for batch_x, batch_y in self.valid_gen:
            y_pred = self.model.predict(batch_x)
            psnr = tf.image.psnr(y_true=batch_y, y_pred=y_pred, max_val=255.0)
            psnr_values.append(psnr.numpy())
        mean_psnr = np.mean(psnr_values)
        logs["val_psnr"] = mean_psnr

        # Menyimpan model jika PSNR terbaik
        if mean_psnr > self.best_psnr:
            self.best_psnr = mean_psnr
            self.model.save(self.model_path)
            print(f" - Saved model with val_psnr: {mean_psnr:.2f} at epoch {epoch+1}")

        print(f' - val_psnr: {mean_psnr:.2f}')

# Fungsi untuk melatih model
def train_model(model, train_gen, valid_gen, steps_per_epoch, validation_steps, epochs, model_path):
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

    # Callback untuk menyimpan model terbaik berdasarkan validation loss
    checkpoint_callback = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    
    # Callback untuk PSNR
    psnr_callback = PSNRCallback(valid_gen, model_path)

    # Melatih model
    history = model.fit(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=valid_gen,
                        validation_steps=validation_steps,
                        callbacks=[checkpoint_callback, psnr_callback])

    return history

# Contoh penggunaan
if __name__ == "__main__":
    # Definisi path untuk menyimpan model
    model_path = "resnet_sr_model.h5"

    # Definisi model (ganti dengan model ResNet atau model yang Anda gunakan)
    model = Sequential([
        Conv2D(64, kernel_size=3, activation='relu', padding='same', input_shape=(256, 256, 3)),
        Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        Conv2D(3, kernel_size=3, activation='sigmoid', padding='same')
    ])

    # Definisi generator data latih dan validasi (ganti dengan generator yang sesuai)
    train_gen = ...  # Definisi generator data latih
    valid_gen = ...  # Definisi generator data validasi

    # Definisi langkah per epoch dan jumlah validation steps
    steps_per_epoch = 100  # Ganti sesuai dengan kebutuhan
    validation_steps = 50  # Ganti sesuai dengan kebutuhan

    # Melatih model
    history = train_model(model, train_gen, valid_gen, steps_per_epoch, validation_steps, epochs=10, model_path=model_path)

    # Tampilkan nilai PSNR terbaik
    print(f"Best PSNR: {psnr_callback.best_psnr:.2f}")

    # Anda juga dapat menggunakan model_path untuk memuat model yang telah disimpan
    # loaded_model = tf.keras.models.load_model(model_path)
