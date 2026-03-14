import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Bidirectional

# Load dataset
df = pd.read_csv("data/combined_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Filter one coin (Ethereum for now)
df = df[df["cryptocurrency"] == "Ethereum"]
df = df.sort_values("timestamp")

prices = df["current_price_usd"].values.reshape(-1, 1)

# Scale
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

sequence_length = 10

X = []
y = []

for i in range(sequence_length, len(scaled_prices)):
    X.append(scaled_prices[i-sequence_length:i])
    y.append(scaled_prices[i])

X = np.array(X)
y = np.array(y)

# Build Model
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True),
                        input_shape=(X.shape[1], 1)))
model.add(GRU(50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

# Train
model.fit(X, y, epochs=15, batch_size=16)

# Save model
model.save("price_model.h5")

# Save scaler
import joblib
joblib.dump(scaler, "scaler.save")

print("Model training complete and saved.")
