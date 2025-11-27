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
y = data.get_column("Price")

# %%
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = data.select(melbourne_features)
X.describe()

# %%
X.head()
# %%
#
# Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# Fit: Capture patterns from provided data. This is the heart of modeling.
# Predict: Just what it sounds like
# Evaluate: Determine how accurate the model's predictions are.

from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

# %%
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

# %%
print(y.head())
