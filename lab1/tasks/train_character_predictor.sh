#!/bin/sh
pipenv run python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "resnet", "train_args": {"batch_size": 256, "epoch": 30}}'
