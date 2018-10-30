#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:47:25 2018

@author: julius
"""

import flask
from flask import request

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.data import Dataset
import math

import argparse
import json

import load_model


def my_input_fn(features):
    """Normalize and pass features to linear or nn classifier for prediction.
  
    Args:
        features: A pandas DataFrame of features
      
    Returns:
        Normalized features of input.
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
        features: pandas DataFrame of features.
      
    Returns:
        pred_class_id: A 'list' of predicted label as 'int'.
        probabilities: A 'list' of prediction probabilities as 'float32'.
    """
   
    prediction_input_fn = lambda: my_input_fn(features)
    
    predictions = list(classifier.predict(input_fn = prediction_input_fn))
    pred_class_id = [int(item['class_ids'][0]) for item in predictions]
    probabilities = [item['probabilities'][item['class_ids'][0]] for item in predictions]

    return pred_class_id, probabilities


def launch_api(classifier_name, host, port):
    '''Launch the api for predictions with a certain classifier.
    
    Args:
        classifier: A trained linear or nn classifier object for predictions.
        host: A 'str', the host url of api.
        port: An 'int', the port of the host api used.
    '''
    
    # Choose which classifier to use.
    if classifier_name == 'dnn_classifier':
        classifier = load_model.load_DNNClassifier()
    elif classifier_name == 'linear_classifier':
        classifier = load_model.load_LinearClassifier()
    else:
        return print('No model matched. Choose one between \'dnn_classifier\' and \'linear_classifier\'.')

    # Launch api.
    app = flask.Flask(__name__)
    app.config["DEBUG"] = True

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    @app.route('/', methods = ['GET'])
    def home():
        return '''
                <h1>Student Identity Judgement</h1>
                <p>An API for judging student identity of authors.</p>
                <p>POST to '/judge' with the following parameters in JSON format: 
                    <br>&emsp;<b>pc</b>: total number of publications
                    <br>&emsp;<b>cn</b>: total number of citations
                    <br>&emsp;<b>hi</b>: h-index
                    <br>&emsp;<b>gi</b>: g-index
                    <br>&emsp;<b>year_range</b>: time range from the first to the last publication
                    <br>&emsp;<b>id</b> <i>(optional)</i>: id of authors</p>
                '''
        
    @app.errorhandler(404)
    def page_not_found(e):
        return '<h1>404</h1><p>The resource could not be found.</p>', 404

    @app.route('/judge', methods = ['POST'])
    def judge():
        
        request_data = request.get_json()
        
        try:
            features = pd.read_json(json.dumps(request_data))
        except:
            return '''
                <h1>Incomplete Query Parameters</h1>
                <p>Five parameters are needed for judging student identity, 
                    at least one is missing.</p>
                '''
        
        labels, probabilities = parse(classifier, features)
        
        # Integrate different DataFrames into one called results_df.
        results_df = pd.DataFrame()
        
        # If id available.
        try:
            results_df['id'] = features['id']
        except:
            pass
        
        results_df['label'] = pd.Series(labels)
        results_df['probability'] = pd.Series(probabilities)
        
        results = json.dumps(json.loads(
                                    results_df.to_json(orient = 'records')),
                             indent = 4)
        
        return results
    
    app.run(host = host, port = port)
    
    
if __name__ == '__main__':
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # API Settings.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--classifier', type = str, default = 'dnn_classifier', 
                        help = 'Choose which classifier to use, dnn_classifier or linear_classifier, default dnn_classifier.')
    parser.add_argument('--host', type = str, default = '127.0.0.1', 
                        help = 'host url of api, default localhost.')
    parser.add_argument('--port', type = int, default = 5000, 
                        help = 'port of the host api used, defulat 5000.')
    
    args = parser.parse_args()
    
    # Launch api.
    launch_api(classifier_name = args.classifier, host = args.host, port = args.port)
    