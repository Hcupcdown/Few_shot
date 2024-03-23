import torch

from .sep_train import SepTester, SepTrainer


def build_trainer(args, model, data):
    return SepTrainer(model, data, args)

def build_tester(args, model, data):
    return SepTester(model, data, args)
