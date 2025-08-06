# retrain_pipeline.py

import sqlite3
import pandas as pd
import joblib
import mlflow
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

def run_retraining():

    df_record_id = pd.read_csv('last_record_id_trained.csv')

    # Get the record id of the last record from the DB, which was used in deployed model training
    
    record_id = df_record_id.iloc[0,0]
    
    # Load data
    conn = sqlite3.connect("/predictions.db")
    df = pd.read_sql(f"SELECT * FROM predictions where id > {record_id} order by id desc", conn)
    conn.close()

    #print(df.iloc[0,0])

    # DataFrames with training dataset
    if len(df) < 50:
        print("Count of new records are less than 50 so, retraining not required.")
        sys.exit(100)
        
    
    X_already_train = pd.read_csv('/data/X_train.csv')
    y_already_train = pd.read_csv('/data/y_train.csv')
    
    # Renaming all the predictions table columns according to the training dataset
    df.rename(columns={
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width": "petal width (cm)",
    "prediction": "species"
    }, inplace=True)

    # Features and label from predicitons table
    
    X_logged = df[[
        "sepal length (cm)", "sepal width (cm)",
        "petal length (cm)", "petal width (cm)"
    ]]
    y_logged = df[["species"]]

    # Concatenating the input data with the training dataset
    
    X = pd.concat([X_already_train,X_logged],ignore_index=True)
    y = pd.concat([y_already_train,y_logged],ignore_index=True)

    # Concatenatiing the entire X and y dataframe
    df_dataset = pd.concat([X,y],axis=1,ignore_index=True)

    # Dropping the duplicate records
    
    df_dataset.drop_duplicates()

    X = df_dataset.iloc[:, :-1]  # All rows, all columns except the last
    y = df_dataset.iloc[:, -1]   # All rows, only the last column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, shuffle = True)

    reference_data = X_already_train
    current_data = X_logged
    features_to_monitor = [col for col in reference_data.columns]# if col != 'species']
    drifted_features_count = 0
    drift_details = {}

    for feature in features_to_monitor:
        ref_mean = reference_data[feature].mean()
        curr_mean = current_data[feature].mean()
        ref_median = reference_data[feature].median()
        curr_median = current_data[feature].median()
        ref_std = reference_data[feature].std()
        curr_std = current_data[feature].std()

        mean_diff_percent = abs((curr_mean - ref_mean) / ref_mean) if ref_mean != 0 else (abs(curr_mean) if curr_mean != 0 else 0)
        median_diff_abs = abs(curr_median - ref_median)
        std_diff_percent = abs((curr_std - ref_std) / ref_std) if ref_std != 0 else (abs(curr_std) if curr_std != 0 else 0)

        feature_drifted = False
        reasons = []
        if mean_diff_percent > 0.10:
            feature_drifted = True
            reasons.append(f"Mean changed by {mean_diff_percent:.2%} (Ref: {ref_mean:.2f}, Curr: {curr_mean:.2f})")
        if median_diff_abs > 0.5:
            feature_drifted = True
            reasons.append(f"Median changed by {median_diff_abs:.2f} (Ref: {ref_median:.2f}, Curr: {curr_median:.2f})")
        if std_diff_percent > 0.15:
            feature_drifted = True
            reasons.append(f"Std Dev changed by {std_diff_percent:.2%} (Ref: {ref_std:.2f}, Curr: {curr_std:.2f})")

        if feature_drifted:
            drifted_features_count += 1
            drift_details[feature] = reasons
            print(f"Drift detected for feature '{feature}': {', '.join(reasons)}")
        else:
            print(f"No significant drift for feature '{feature}'.")

    print(f"\nTotal features with detected drift: {drifted_features_count}")
    
    if drifted_features_count>0:
        
        mlflow.set_experiment("Iris Classification Experiment")
        mlflow.autolog()
        with mlflow.start_run(run_name="RandomForest_Iris"):
            # Log parameters
            n_estimators = 100
            max_depth = 10
            model_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model_rf.fit(X_train, y_train)
            y_pred_rf = model_rf.predict(X_test)
            
            # Log metrics
            accuracy_rf = accuracy_score(y_test, y_pred_rf)
            precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
            recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
            f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

            mlflow.log_metrics({
               "testing_accuracy": accuracy_rf,
               "testing_precision": precision_rf,
               "testing_recall": recall_rf,
               "testing_f1_score": f1_rf
            })

            # Register to MLflow
            mlflow.register_model(
             f"runs:/{mlflow.active_run().info.run_id}/model",
             "IrisSpeciesClassifier"
            )
            
        with open('last_record_id_trained.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['RECORDID'])
            writer.writerow([df.iloc[0,0]])
            
        print("New model trained and logged.")
        sys.exit(0)
    else:
        print("No data drift. Skipping retraining.")
        sys.exit(100)