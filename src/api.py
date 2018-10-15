#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:47:25 2018

@author: julius
"""

import flask
from flask import request, jsonify

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.data import Dataset
import math

import argparse

import load_model


def my_input_fn(features):
    """Normalize and pass features to linear or nn classifier for prediction.
  
    Args:
        features: A pandas DataFrame of features
      
    Returns:
        Normalized feature of input.
    """

    # 53 is the max year_range in training set.
    max_year_range = 53

    normalized_features = pd.DataFrame()
    
    for feature in ['pc', 'cn', 'hi', 'gi']:
        normalized_features[feature] = features[feature].apply(
                lambda val: math.log(val + 1.0))   

    normalized_features['year_range'] = features['year_range'].apply(
            lambda val: val / max_year_range)
        
    features = {key: np.array(value) for key, value in dict(normalized_features).items()}
    
    ds = Dataset.from_tensor_slices(features)
    ds = ds.batch(1).repeat(1)
        
    features = ds.make_one_shot_iterator().get_next()
    
    return features


def parse(classifier, features):
    """Predict student identity for an author.
  
    Args:
        classifier: a trained linear or nn classifier object for predictions.
        features: pandas DataFrame of features
      
    Returns:
        An 'int' of predicted label
    """
   
    prediction_input_fn = lambda: my_input_fn(features)
    
    predictions = classifier.predict(input_fn = prediction_input_fn)
    pred_class_id = np.array([item['class_ids'][0] for item in predictions])[0]

    return int(pred_class_id)


def launch_api(classifier_name, host, port):
    '''Launch the api for predictions with a certain classifier.
    
    Args:
        classifier: A trained linear or nn classifier object for predictions.
        host: A 'str', the host url of api.
        port: An 'int', the port of the host api used.
    '''
    
    if classifier_name == 'dnn_classifier':
        classifier = load_model.load_DNNClassifier()
    elif classifier_name == 'linear_classifier':
        classifier = load_model.load_LinearClassifier()
    else:
        return print('No model matched. Choose one between \'dnn_classifier\' and \'linear_classifier\'.')

    api = flask.Flask(__name__)
    api.config["DEBUG"] = True

    @api.route('/', methods = ['GET'])
    def home():
        return '''
                <h1>Student Identity Judgement</h1>
                <p>A prototype API for judging student identity of authors.</p>
                <p>Go to '/judge' with the following query parameters: 
                    <br>&emsp;pc, total number of publications
                    <br>&emsp;cn, total number of citations
                    <br>&emsp;hi, h-index
                    <br>&emsp;gi, g-index
                    <br>&emsp;year_range, time range from the first to the last publication</p>
                '''
        
    @api.errorhandler(404)
    def page_not_found(e):
        return '<h1>404</h1><p>The resource could not be found.</p>', 404

    @api.route('/judge', methods = ['GET'])
    def judge():
        
        features = pd.DataFrame(columns = ['pc', 'cn', 'hi','gi', 'year_range'])
        
        query_parameters = request.args
        
        for feature in ['pc', 'cn', 'hi', 'gi', 'year_range']:
            if feature in query_parameters:
                features[feature] = pd.Series(int(query_parameters.get(feature)))
            else:
                return '''
                        <h1>Incomplete Query Parameters</h1>
                        <p>Five parameters are needed for judging student identity, 
                            at least one is missing.</p>
                        '''
        
        label = parse(classifier, features)
    
        features = {key: int(value) for key, value in dict(features).items()}
        result = {'features': features, 'label': label}
        
        return jsonify(result)
    
    api.run(host = host, port = port)
    
    
if __name__ == '__main__':
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # API Settings.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--classifier', type = str, default = 'dnn_classifier', 
                        help = 'Choose which classifier to use, dnn_classifier or linear_classifier.')
    parser.add_argument('--host', type = str, default = '127.0.0.1', 
                        help = 'host url of api, default localhost.')
    parser.add_argument('--port', type = int, default = 5000, 
                        help = 'port of the host api used, defulat 5000.')
    
    args = parser.parse_args()
    
    # Launch api.
    launch_api(classifier_name = args.classifier, host = args.host, port = args.port)
    