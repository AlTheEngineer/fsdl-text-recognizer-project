#!/usr/bin/env python
import argparse
import json
import importlib
from typing import Dict
import os

from text_recognizer.character_predictor import CharacterPredictor

from training.util import train_model

"""
Processes config/json files to load the correct objects
"""


def run_evaluation(evaluation_config: Dict, gpu_ind: int, use_wandb: bool=True):
    """
    evaluation_config is of the form
    {
        "dataset": "EmnistLinesDataset",
        "dataset_args": {
            "max_overlap": 0.4
        },
        "model": "LineModel",
        "network": "line_cnn_sliding_window",
    }
    gpu_ind: integer specifying which gpu to use
    """
    print(f'Running evaluation with config {evaluation_config} on GPU {gpu_ind}')

    datasets_module = importlib.import_module('text_recognizer.datasets')
    dataset_class_ = getattr(datasets_module, evaluation_config['dataset'])
    dataset_args = evaluation_config.get('dataset_args', {})
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)
    
    models_module = importlib.import_module('text_recognizer.models')
    # CharacterModel, LineModel...etc.
    model_class_ = getattr(models_module, evaluation_config['model'])

    networks_module = importlib.import_module('text_recognizer.networks')
    # mlp, resnet...etc.
    network_fn_ = getattr(networks_module, evaluation_config['network'])
    network_args = evaluation_config.get('network_args', {})
    # model object
    model = model_class_(dataset_cls=dataset_class_, network_fn=network_fn_, dataset_args=dataset_args, network_args=network_args)
    print(model)
    
    model.load_weights()
    
    train_score = model.evaluate(dataset.x_train, dataset.y_train)
    
    print(f'Training evaluation: {train_score}')
    
    score = model.evaluate(dataset.x_test, dataset.y_test)
    
    print(f'Test evaluation: {score}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Provide index of GPU to use."
    )
    parser.add_argument(
        "evaluation_config",
        type=str,
        help="JSON of experiment to run (e.g. '{\"dataset\": \"EmnistDataset\", \"model\": \"CharacterModel\", \"network\": \"mlp\"}'"
    )
    args = parser.parse_args()

    evaluation_config = json.loads(args.evaluation_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'
    run_evaluation(evaluation_config, args.gpu)

