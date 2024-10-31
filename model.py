import os
import neuralnet as nn
from utils.common import exists
import tensorflow as tf
import numpy as np

class logger:
    def __init__(self, path, values) -> None:
        self.path = path
        self.values = values

class SRCNN:
    def __init__(self, architecture="915"):
        if architecture == "915":
            self.model = nn.SRCNN915()
        elif architecture == "935":
            self.model = nn.SRCNN935()
        elif architecture == "955":
            self.model = nn.SRCNN955()
        elif architecture == "18210":
            self.model = nn.SRCNN18210()
        elif architecture == "18610":
            self.model = nn.SRCNN18610()
        elif architecture == "181010":
            self.model = nn.SRCNN181010()
        elif architecture == "1895210":
            self.model = nn.SRCNN1895210()
        elif architecture == "27315":
            self.model = nn.SRCNN27315()
        elif architecture == "27915":
            self.model = nn.SRCNN27915()
        elif architecture == "271515":
            self.model = nn.SRCNN271515()
        elif architecture == "36420":
            self.model = nn.SRCNN36420()
        elif architecture == "361220":
            self.model = nn.SRCNN361220()
        elif architecture == "362020":
            self.model = nn.SRCNN362020()
        elif architecture == "97315":
            self.model = nn.SRCNN97315()
        elif architecture == "9315":
            self.model = nn.SRCNN9315()
        elif architecture == "973135":
            self.model = nn.SRCNN973135()
        else:
            raise ValueError("\"architecture\" must be 915, 935, 955, 18210, 18610, 181010, 1895210, 27315, 27915, 271515, 36420, 361220, 362020, 97315, 9315 or 973135")

        self.optimizer = None
        self.loss =  None
        self.metric = None
        self.model_path = None
        self.ckpt = None
        self.ckpt_dir = None
        self.ckpt_man = None
    
    def setup(self, optimizer, loss, metric, model_path):
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.model_path = model_path
    
    def load_checkpoint(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), 
                                        optimizer=self.optimizer,
                                        net=self.model)
        self.ckpt_man = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=1)
        self.ckpt.restore(self.ckpt_man.latest_checkpoint)
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def predict(self, lr):
        sr = self.model(lr)
        return sr
    
    def evaluate(self, dataset, batch_size=64):
        losses, metrics = [], []
        isEnd = False
        while not isEnd:
            lr, hr, isEnd = dataset.get_batch(batch_size, shuffle_each_epoch=False)
            sr = self.predict(lr)
            losses.append(self.loss(hr, sr))
            metrics.append(self.metric(hr, sr))

        metric = tf.reduce_mean(metrics).numpy()
        loss = tf.reduce_mean(losses).numpy()
        return loss, metric

    def train(self, train_set, valid_set, batch_size, steps, save_every=1,
              save_best_only=False, save_log=False, log_dir=None, phase=1,
              cumulative_psnr=None):

        if save_log and log_dir is None:
            raise ValueError("log_dir must be specified if save_log is True")
        os.makedirs(log_dir, exist_ok=True)
        dict_logger = {"loss":       logger(path=os.path.join(log_dir, "losses.npy"),      values=[]),
                       "metric":     logger(path=os.path.join(log_dir, "metrics.npy"),     values=[]),
                       "val_loss":   logger(path=os.path.join(log_dir, "val_losses.npy"),  values=[]),
                       "val_metric": logger(path=os.path.join(log_dir, "val_metrics.npy"), values=[])}

        for key in dict_logger.keys():
            path = dict_logger[key].path
            if exists(path):
                dict_logger[key].values = np.load(path).tolist()

        cur_step = self.ckpt.step.numpy()
        max_steps = steps + self.ckpt.step.numpy()

        prev_loss = np.inf
        if save_best_only and exists(self.model_path):
            self.load_weights(self.model_path)
            prev_loss, _ = self.evaluate(valid_set)
            self.load_checkpoint(self.ckpt_dir)

        loss_buffer = []
        metric_buffer = []
        while cur_step < max_steps:
            cur_step += 1
            self.ckpt.step.assign_add(1)
            lr, hr, _ = train_set.get_batch(batch_size)
            loss, metric = self.train_step(lr, hr)
            loss_buffer.append(loss)
            metric_buffer.append(metric)

            if cur_step % save_every == 0 or cur_step >= max_steps:
                loss, metric = tf.reduce_mean(loss_buffer), tf.reduce_mean(metric_buffer)
                loss_buffer, metric_buffer = [], []
                val_loss, val_metric = self.evaluate(valid_set)
                print(f"phase: {phase} - step: {cur_step}/{max_steps} - loss: {loss:.5f} - psnr: {metric:.5f} - val_loss: {val_loss:.5f} - val_psnr: {val_metric:.5f}")
                
                dict_logger["loss"].values.append(loss)
                dict_logger["metric"].values.append(metric)
                dict_logger["val_loss"].values.append(val_loss)
                dict_logger["val_metric"].values.append(val_metric)

                if (save_best_only and val_loss < prev_loss) or not save_best_only:
                    prev_loss = val_loss
                    self.save_weights(self.model_path)
                    self.ckpt_man.save()

                if save_log:
                    for key in dict_logger.keys():
                        path = dict_logger[key].path
                        np.save(path, dict_logger[key].values)
        
        average_psnr = np.mean(dict_logger["val_metric"].values)

        if cumulative_psnr is not None:
            cumulative_psnr = (cumulative_psnr + average_psnr) / 2
            print(f"Cumulative PSNR at phase {phase}, step {cur_step}: {cumulative_psnr}")
            return cumulative_psnr
        else:
            return average_psnr

    @tf.function    
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            sr = self.model(lr, training=True)
            loss = self.loss(hr, sr)
            metric = self.metric(hr, sr)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, metric
