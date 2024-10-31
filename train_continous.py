import os
import argparse
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from utils.dataset import dataset
from utils.common import PSNR
from model import SRCNN

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

parser = argparse.ArgumentParser()
parser.add_argument("--steps1",          type=int, default=160000, help='Steps for first training phase')
parser.add_argument("--steps2",          type=int, default=160000, help='Steps for second training phase')
#parser.add_argument("--steps3",          type=int, default=20000, help='Steps for second training phase')
#parser.add_argument("--steps4",          type=int, default=20000, help='Steps for second training phase')
#parser.add_argument("--steps5",          type=int, default=20000, help='Steps for second training phase')
parser.add_argument("--batch-size",      type=int, default=1,    help='Batch size for training')
parser.add_argument("--architecture",    type=str, default="27315",  help='Architecture type (915, 935, 955)')
parser.add_argument("--save-every",      type=int, default=10000,   help='Save model every N steps')
parser.add_argument("--save-best-only",  type=int, default=0,      help='Save only the best model')
parser.add_argument("--save-log",        type=int, default=1,      help='Save log')
parser.add_argument("--ckpt-dir",        type=str, default="checkpoint/SRCNN27315v2",     help='Checkpoint directory')

FLAGS, unparsed = parser.parse_known_args()
steps1 = FLAGS.steps1
steps2 = FLAGS.steps2
batch_size = FLAGS.batch_size
save_every = FLAGS.save_every
save_log = (FLAGS.save_log == 1)
save_best_only = (FLAGS.save_best_only == 1)

architecture = FLAGS.architecture
if architecture not in ["915", "935", "955", "18210", "18610", "181010", "27315", "27915", "271515", "97315", "9315"]:
    raise ValueError("architecture must be 915, 935, 955, 18210, 18610, 181010, 27315, 27915, 271515, 97315 or 9315")

ckpt_dir = FLAGS.ckpt_dir
if not ckpt_dir or ckpt_dir == "default":
    ckpt_dir = f"checkpoint/SRCNN{architecture}"
model_path = os.path.join(ckpt_dir, f"SRCNN-{architecture}.h5")

# Function to initialize dataset
def init_dataset(dataset_dir, subset, lr_crop_size, hr_crop_size):
    dataset_instance = dataset(dataset_dir, subset)
    dataset_instance.generate(lr_crop_size, hr_crop_size)
    dataset_instance.load_data()
    return dataset_instance

lr_crop_size = 33
hr_crop_size = 21
if architecture == "935":
    hr_crop_size = 19
elif architecture == "955":
    hr_crop_size = 17
elif architecture == "18210" or architecture == "18610" or architecture == "181010":
    hr_crop_size = 33
elif architecture == "27315" or architecture == "27915" or architecture == "271515":
    hr_crop_size = 33  # sesuaikan ukuran potongan gambar yang diperlukan
elif architecture == "97315":
    hr_crop_size = 33  # sesuaikan ukuran potongan gambar yang diperlukan
elif architecture == "9315":
    hr_crop_size = 33  # sesuaikan ukuran potongan gambar yang diperlukan

def train_model(model, steps, dataset_dir):
    train_set = init_dataset(dataset_dir, "train", lr_crop_size, hr_crop_size)
    valid_set = init_dataset(dataset_dir, "validation", lr_crop_size, hr_crop_size)
    model.train(train_set, valid_set, steps=steps, batch_size=batch_size,
                save_best_only=save_best_only, save_every=save_every,
                save_log=save_log, log_dir=ckpt_dir)

def main():
    model = SRCNN(architecture)
    model.setup(optimizer=Adam(learning_rate=2e-5),
                loss=MeanSquaredError(),
                model_path=model_path,
                metric=PSNR)
    
    model.load_checkpoint(ckpt_dir)

    # First phase of training
    train_model(model, steps1, "dataset1")
    
    # Second phase of training
    train_model(model, steps2, "dataset2")

    # 3 phase of training
    #train_model(model, steps1, "dataset3")
    
    # 4 phase of training
    #train_model(model, steps2, "dataset4")

    # 5 phase of training
    #train_model(model, steps2, "dataset5")

if __name__ == "__main__":
    main()
