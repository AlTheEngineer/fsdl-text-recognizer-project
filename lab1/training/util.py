from time import time
from typing import Callable, Optional, Union, Tuple

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import RMSprop

from text_recognizer.datasets.base import Dataset
from text_recognizer.models.base import Model
from training.gpu_util_sampler import GPUUtilizationSampler


EARLY_STOPPING = False
# custom written for this lab: checks gpu utilization after each batch training
GPU_UTIL_SAMPLER = True


def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int, gpu_ind: Optional[int]=None, use_wandb=False) -> Model:
    callbacks = []
    # Pass early stopping to callback
    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)
    # Monitor GPU utilization of the GPU in use
    if GPU_UTIL_SAMPLER and gpu_ind is not None:
        gpu_utilization = GPUUtilizationSampler(gpu_ind)
        callbacks.append(gpu_utilization)

    # Print model structure
    model.network.summary()
    # Start timer
    t = time()
    # Train model on dataset
    history = model.fit(dataset, batch_size, epochs, callbacks)
    # Print training time
    print('Training took {:2f} s'.format(time() - t))
    # Print GPU utilization if True
    if GPU_UTIL_SAMPLER and gpu_ind is not None:
        gpu_utilizations = gpu_utilization.samples
        print(f'GPU utilization: {round(np.mean(gpu_utilizations), 2)} +- {round(np.std(gpu_utilizations), 2)}')
    # Return trained model
    return model

