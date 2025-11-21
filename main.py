# %%
import polars as pl


def load_data():
    melbourne_file_path = "datasets/melb_data.csv"
    df = pl.read_csv(melbourne_file_path)
    return df


# %%
print("Hello from mlexps!")
data = load_data()
# %%
data.drop_nulls()
data.describe()
