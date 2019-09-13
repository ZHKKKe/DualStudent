import numpy as np

from third_party.mean_teacher import ramps as mt_ramps


def zero_cosine_rampdown(current, epochs):
    return float(.5 * (1.0 + np.cos((current - 1) * np.pi / epochs)))


def sigmoid_rampup(current, rampup_length):
    return mt_ramps.sigmoid_rampup(current, rampup_length)


def linear_rampup(current, rampup_length):
    return mt_ramps.linear_rampup(current, rampup_length)


def cosine_rampdown(current, rampdown_length):
    return mt_ramps.cosine_rampdown(current, rampdown_length)
