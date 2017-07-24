#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
tensorflow dropout
"""

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#加载数据
digits