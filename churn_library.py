'''
Function library to analyze customer churn.

Author: Ryan Gordon
Creation Date: 08/21/2023
'''

import os
import time
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap

# Useful path variables for saving files
PATH_IMAGE_RESULTS = './images/results/'
PATH_IMAGE_EDA = './images/eda/'
PATH_MODELS = './models/'

os.environ['QT_QPA_PLATFORM']='offscreen'
sns.set()

def import_data(pth = './data/bank_data.csv'):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df, image_fig_dir = PATH_IMAGE_EDA):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Figure: Histogram of Customer_Age
    plt.figure()
    plt.hist(df['Customer_Age'])
    plt.xlabel('Customer Age')
    plt.ylabel('Count')
    plt.title('Histogram of Customer Age')
    plt.tight_layout()
    pth_histogram_age = os.path.join(image_fig_dir, 'Customer_Age_Histrogram.png')
    plt.savefig(pth_histogram_age)
    plt.close()
    # Figure: Histogram of Gender
    plt.figure()
    plt.hist(df['Gender'])
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Histogram of Gender')
    plt.tight_layout()
    pth_histogram_gender = os.path.join(image_fig_dir, 'Customer_Gender_Histrogram.png')
    plt.savefig(pth_histogram_gender)
    plt.close()
    # Figure: Customer_Age vs. Dependent_count
    plt.figure()
    plt.plot(df['Dependent_count'],df['Customer_Age'],'ro')
    plt.xlabel('Dependent Count')
    plt.ylabel('Customer Age')
    plt.title('Customer Age vs. Dependent Count')
    plt.tight_layout()
    pth_plot_age_vs_dependents = os.path.join(image_fig_dir, 'Customer_Age_vs_Depenent_Count.png')
    plt.savefig(pth_plot_age_vs_dependents)
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
            for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    # creating "Churn" feature using Attrition flag
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    for category in category_lst:
        values_lst = []
        groups = df.groupby(category).mean()['Churn']
        for val in df[category]:
            values_lst.append(groups.loc[val])
        if response:
            df[category + '_' + response] = values_lst
        else:
            df[category] = values_lst
        df[f'{category}_{response}'] = values_lst
    return df


def perform_feature_engineering(df, response, test_size = 0.3, random_state = 42):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index y column]
              test_size: determines the fraction of the data to be used for the testing
              set in train_test_split()
              random_state: used to control the degree of randomization in train_test_split()

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Get a list of categorical columns in df
    categorical_cols = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    
    # Use encoder_helper() to turn categorical columns into new ones with proportion of churn
    df = encoder_helper(df, categorical_cols, response)
    X = pd.DataFrame()
    y = df['Churn']
    cols_to_keep = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X[cols_to_keep] = df[cols_to_keep]
    return train_test_split(X, y, test_size = test_size, random_state = random_state)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_pth = './images/results/classification_results.png'):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            output_pth: path to where classification report image will be saved

    output:
             None
    '''
    # Print classification reports for Random Forest and Logistic Regression models
    print('Random Forest Classifier Testing Results:')
    print(classification_report(y_test, y_test_preds_rf))
    print('Random Forest Classifier Training Results:')
    print(classification_report(y_train, y_train_preds_rf))
    print('Logistic Regression Testing Results:')
    print(classification_report(y_test, y_test_preds_lr))
    print('Logistic Regression Training Results:')
    print(classification_report(y_train, y_train_preds_lr))
    
    # Create plot summarizing text from classification reports
    plt.figure(figsize=(5, 9))
    plt.text(0.01, 1.00, 'Random Forest Test Results', {'fontsize': 12},
             fontproperties='monospace')
    plt.text(0.01, 0.775, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.725, 'Random Forest Train Results', {'fontsize': 12},
             fontproperties='monospace')
    plt.text(0.01, 0.5, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.450, 'Logistic Regression Test Results', {'fontsize': 12},
             fontproperties='monospace')
    plt.text(0.01, 0.225, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.175, 'Logistic Regression Train Results', {'fontsize': 12},
             fontproperties='monospace')
    plt.text(0.01, -0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


def feature_importance_plot(model, X_data, output_pth = './images/results/feature_importances.png'):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    
    # Calculate the importance of each feature for the given model
    feature_importances = model.feature_importances_
    # Sort the features in feature_importance by descending order
    order_descending = np.argsort(feature_importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    cols_descending = [X_data.columns[n] for n in order_descending]
    # Create figure of feature importances in descending order
    plt.figure()
    plt.bar(range(X_data.shape[1]), feature_importances[order_descending])
    plt.xticks(range(X_data.shape[1]), cols_descending, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()

    
def train_models(X_train, X_test, y_train, y_test, 
                 grid_params_rfc = {'n_estimators': [100, 500],
                     'max_features': ['auto', 'sqrt'],
                     'max_depth': [1, 10, 100],
                     'criterion': ['gini', 'entropy']},
                pth_image_results = PATH_IMAGE_RESULTS,
                pth_model = PATH_MODELS):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              grid_params_rfc: dictionary of parameters for random forest grid search
    output:
              None
    '''
    
    # Instances of models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    # Cross validation for random forest; default cv = 5
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=grid_params_rfc, cv=5)
    # Training the random forest classifier model, timing it, and printing the totoal time
    start_time = time.time()
    print("Training the random forest classifier model")
    cv_rfc.fit(X_train, y_train)
    # Calculating total time to run random forest classifier
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print("Training the random forest classifier took {} \ and minutes and {} seconds.".format(minutes, seconds))
    # Training the random forest classifier model, timing it, and printing the totoal time
    start_time = time.time()
    print("Training the logistic regression model")
    lrc.fit(X_train, y_train)
    # Calculating total time to run logistic regression model
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print("Training the logistic regression model took {} and minutes and {} seconds.".format(minutes, seconds))
    # Predictions for random forest
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    # predictions for logistic regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    # Make ROC curve plot for both models and save
    plt.figure()
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.tight_layout()
    roc_file_pth = os.path.join(pth_image_results, 'roc_curves.png')             
    plt.savefig(roc_file_pth, bbox_inches='tight')
    plt.close()
    # Explainer for model output
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    print("explainer created")
    shap_values = explainer.shap_values(X_test)
    print("shap_values created")
    # Summary figure
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    # Capture the current figure after shap.summary_plot
    summary_file_pth = os.path.join(pth_image_results, 'summary.png')
    fig.savefig(summary_file_pth, bbox_inches='tight')
    plt.close(fig)
    # Create and save figure for classification report
    pth_classification_results = os.path.join(pth_image_results, 'classification_results.png')
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_pth=pth_classification_results)
    # Save the best models
    pth_rfc_model = os.path.join(pth_model, 'rfc_model.pkl')
    joblib.dump(cv_rfc.best_estimator_, pth_rfc_model)
    print("Saved Random Forest Classifier model to {}".format(pth_rfc_model))
    pth_lr_model = os.path.join(pth_model, 'logistic_model.pkl')
    joblib.dump(lrc, pth_lr_model)
    print("Saved Logistic Regression model to {}".format(pth_lr_model))
    # Create & store feature importances
    feature_importance_pth = os.path.join(pth_image_results, 'feature_importances.png')
    feature_importance_plot(cv_rfc.best_estimator_, X_train, output_pth = feature_importance_pth)

        
if __name__ == "__main__":
    PATH_TO_DATA = r"./data/bank_data.csv"
    df = import_data(PATH_TO_DATA)
    perform_eda(df)
    response = 'Churn'
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, response)
    train_models(X_train, X_test, y_train, y_test)