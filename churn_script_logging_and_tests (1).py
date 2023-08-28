'''
The script that contains test functions to be run with `pytest`.

Author: Ryan Gordon
Creation Date: 08/24/2023
'''

import os
import logging
import pytest
import churn_library as cls


# Configure log file
logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def assert_true(condition, success_msg, fail_msg):
    '''
    Assertion to test the truth value of a given condition, used for error logging.
        input:
            condition: The condition to be tested
            success_msg: message to show when assertion is successful
            fail_msg: message to show when assertion fails
        output:
             None
    '''
    try:
        assert condition
        logging.info(success_msg)
    except AssertionError as err:
        logging.error(fail_msg)
        # err.args requires a set
        err.args = (fail_msg,)
        raise err
        
   
def assert_equal(a, b, success_msg, fail_msg):
    '''
    Equality assertion test comparing a and b.
        input:
            a: first variable for comparison
            b: second variable for comparison
            success_msg: message for case when a and b are equal
            fail_msg: message for case when a and b are not equal

        output:
            None
    '''
    try:
        assert a == b
        logging.info(success_msg)
    except AssertionError as err:
        additional_details = f"{a} != {b}"
        msg = f"{fail_msg} ({additional_details})"
        logging.error(msg)
        # err.args requires a set
        err.args = (msg)
        raise err

        
def remove_all_files(directory):
    '''
    Removes all files in a directory.
        input:
            directory: Directory path
        output:
            None
    '''
    for filename in os.listdir(directory):
        # Creates path to all files in the specified directory by iterating
        file_path = os.path.join(directory, filename)
        # If a file is present, it is removed
        if os.path.isfile(file_path):
            os.remove(file_path)
            

@pytest.fixture(scope='module')

def df():
    '''
    The DataFrame object fixture to use in test functions
    '''
    df = cls.import_data("./data/bank_data.csv")
    return df


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df):
    '''
    test perform eda function
    '''
    image_dir_eda = './test_images/eda/'
    remove_all_files(image_dir_eda)
    cls.perform_eda(df, image_fig_dir = image_dir_eda)
    assert_true(os.path.isfile("./test_images/eda/Customer_Age_Histogram.png"),
                "Age histogram plot created.",
                "Age histogram plot not created.")


def test_encoder_helper(df):
    '''
    test encoder helper
    '''
    response = 'Churn'
    category_lst = ['Gender']
    df = cls.encoder_helper(df, category_lst, response)
    assert_true('Gender_Churn' in df.columns, "Gender_Churn column exists.", "Gender_Churn column does not exist.")


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    response = 'Churn'
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, response, test_size=0.4)
    assert_equal(len(X_train), round(len(df) * (1-test_size)),
                "X_train has the correct number of rows.",
                "X_train does not have the correct number of rows.")
    assert_equal(len(X_test), round(len(df) * test_size),
                "X_test has the correct number of rows.",
                "X_test does not have the correct number of rows.")
    assert_equal(len(y_train), round(len(df) * (1-test_size)),
                "y_train has the correct number of rows.",
                "y_train does not have the correct number of rows.")
    assert_equal(len(y_test), round(len(df) * test_size),
                "y_test has the correct number of rows.",
                "y_test does not have the correct number of rows.")


def test_train_models():
    '''
    test train_models
    '''
    response = 'Churn'
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, response, test_size = 0.4)
    
    # Paths to directories for testing figure and model creation
    test_image_dir = 'test_images/results/'
    test_model_dir = 'test_models/'
    
    # Remove files from the above directories prior to running test
    remove_all_files(image_dir_results)
    remove_all_files(model_dir)

    cls.train_models(X_train, X_test, y_train, y_test,
                     param_grid_rfc={'n_estimators': [200, 500],
                                     'max_features': ['auto'],
                                     'max_depth' : [4],
                                     'criterion' :['gini']},
                     pth_image_results=test_image_dir,
                     pth_model=test_model_dir)
    
    assert_true(os.path.isfile("./test_images/results/classification_results.png"),
                "classification_results.png is created.",
                "classification_results.png is not created.")
    
    assert_true(os.path.isfile("./test_models/rfc_model.pkl"),
                "rfc_model.pkl is created.",
                "rfc_model.pkl is not created.")








