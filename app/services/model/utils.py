import pandas as pd
import app.services.datastore as ds


# Load cleaned data
def load_train_data():
    ds.training_df = pd.read_csv("data/model/train_data_cleaned.csv")


def load_validation_data():
    ds.validation_df = pd.read_csv("data/model/val_data_cleaned.csv")


def load_test_data(test_file):
    test_file = 'data/model/' + test_file + '.csv'
    ds.testing_df = pd.read_csv(test_file)
