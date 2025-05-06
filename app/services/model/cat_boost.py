import app.services.datastore as ds
from app.services.model.utils import load_test_data, load_train_data, load_validation_data
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import logging
from joblib import dump, load

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Pipe-delimited log format: TIME|LEVEL|MODULE|MESSAGE
formatter = logging.Formatter(
    fmt='%(asctime)s|%(levelname)-8s|%(module)-15s|%(funcName)-20s|%(lineno)4d|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Stream handler (console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Avoid adding multiple handlers if reloaded
if not logger.hasHandlers():
    logger.addHandler(console_handler)


def run_catboost_training():
    # ---- Step 1: Define features and utility ----
    numeric_cols = ["WicketDeliveries", "EconomyRate", "StrikeRate", "BoundaryPercentage", "WinLoss"]
    categorical_cols = ["BattingTeam", "Batter", "battingStyles_x", "venue", "Phase", "Bowler"]
    all_features = numeric_cols + categorical_cols
    target_col = "Selected"

    logger.info("Loading test data, training data, validation data")
    # ---- Step 2: Load & clean data ----
    load_train_data()
    load_validation_data()

    logger.info("Cleaning Numerical Data Set")
    # Clean numeric columns
    train_df = clean_numeric(ds.training_df.copy(), numeric_cols)
    val_df = clean_numeric(ds.validation_df.copy(), numeric_cols)

    # Ensure categorical columns are strings
    for col in categorical_cols:
        train_df[col] = train_df[col].astype(str)
        val_df[col] = val_df[col].astype(str)

    # ---- Step 3: Prepare data for CatBoost ----
    logger.info("Preparing Data Set for CatBoost")
    x_train, y_train = train_df[all_features], train_df[target_col]
    x_val, y_val = val_df[all_features], val_df[target_col]

    train_pool = Pool(x_train, label=y_train, cat_features=categorical_cols)
    val_pool = Pool(x_val, label=y_val, cat_features=categorical_cols)

    # ---- Step 4: Train CatBoost model ----
    logger.info("Training CatBoost Model")
    model = CatBoostClassifier(depth=5, random_seed=42, verbose=100)
    model.fit(train_pool, eval_set=val_pool)
    dump(model, "data/model/bowlmate_catboost_model.pkl")

    # ---- Step 5: Evaluate on validation ----
    logger.info("Evaluating CatBoost Model")
    y_val_pred = model.predict(x_val)
    logger.info("📊 Validation Set Performance:")
    logger.info(classification_report(y_val, y_val_pred))
    logger.info(confusion_matrix(y_val, y_val_pred))

    val_results = x_val.copy()
    val_results["Actual"] = y_val
    val_results["Predicted"] = y_val_pred
    val_results.to_csv("data/model/validation_results.csv", index=False)

    # ---- Step 7: Feature Importance ----
    importances = model.get_feature_importance(prettified=True)
    logger.info("📈 Feature Importances:\n%s", importances)


def run_catboost_testing(test_file):
    # ---- Step 1: Define features and utility ----
    numeric_cols = ["WicketDeliveries", "EconomyRate", "StrikeRate", "BoundaryPercentage"]
    categorical_cols = ["BattingTeam", "Batter", "battingStyles_x", "venue", "Phase"]
    all_features = numeric_cols + categorical_cols
    target_col = "Selected"

    logger.info("Loading test data, training data, validation data")
    # ---- Step 2: Load & clean data ----
    load_test_data(test_file)

    logger.info("Cleaning Numerical Data Set")
    # Clean numeric columns
    test_df = clean_numeric(ds.testing_df.copy(), numeric_cols)

    # Ensure categorical columns are strings
    for col in categorical_cols:
        test_df[col] = test_df[col].astype(str)

    # ---- Step 3: Prepare data for CatBoost ----
    logger.info("Preparing Data Set for CatBoost")
    x_test, y_test = test_df[all_features], test_df[target_col]

    test_pool = Pool(x_test, cat_features=categorical_cols)

    # ---- Step 5: Evaluate on test ----
    logger.info("Evaluating on Test Data")
    model = load('data/model/bowlmate_catboost_model.pkl')
    y_test_pred = model.predict(x_test)
    logger.info("📊 Test Set Performance:")
    logger.info(classification_report(y_test, y_test_pred))
    logger.info(confusion_matrix(y_test, y_test_pred))

    # ---- Step 7: Feature Importance ----
    importances = model.get_feature_importance(prettified=True)
    logger.info("📈 Feature Importances:\n%s", importances)

    return {"processed_file": test_file}


def clean_numeric(df, numeric_columns):
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df
