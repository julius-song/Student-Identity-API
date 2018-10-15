#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:17:29 2018

@author: julius
"""

import os

import tensorflow as tf


def construct_feature_columns():
    """Construct the TensorFlow Feature Columns.
    
    Returns:
        A set of feature columns
    """ 
    
    return set([tf.feature_column.numeric_column(my_feature) 
                for my_feature in ['pc', 'cn', 'hi', 'gi', 'year_range']])
 

def load_LinearClassifier(model_dir = 
                          os.path.join(os.getcwd(), 'linear_classifier')):
    '''Load the trained LinearClassifier Model.

    Args:
        model_dir: A 'str', the directory where the trained LinearClassifier
            Model is stored.

    Returns:
        A trained LinearClassifier object.
    '''
    
    linear_classifier = tf.estimator.LinearClassifier(
            feature_columns = construct_feature_columns(),
            model_dir = model_dir)

    return linear_classifier
    
    
def load_DNNClassifier(model_dir = 
                       os.path.join(os.getcwd(), 'dnn_classifier')):
    '''Load the trained DNNClassifier Model.

    Args:
        model_dir: A 'str', the directory where the trained DNNClassifier
            Model is stored.

    Returns:
        A trained DNNClassifier object.
    '''
    
    dnn_classifier = tf.estimator.DNNClassifier(
            hidden_units = [10, 10],
            feature_columns = construct_feature_columns(),
            model_dir = model_dir)
    
    return dnn_classifier

