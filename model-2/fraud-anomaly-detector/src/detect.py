import pandas as pd

# Load historical data
df = pd.read_csv("data/transactions.csv")

# Calculate IQR
Q1 = df["amount"].quantile(0.25)
Q3 = df["amount"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Normal transaction range:")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")
print("-" * 40)

# Real-time user input
while True:
    user_input = input("Enter transaction amount (or 'exit'): ")

    if user_input.lower() == "exit":
        print("Exiting anomaly detector.")
        break

    try:
        amount = float(user_input)
    except ValueError:
        print("❌ Please enter a valid number.")
        continue

    if amount < lower_bound or amount > upper_bound:
        print("🚨 ANOMALY DETECTED (Suspicious Transaction)")
    else:
        print("✅ Normal Transaction")

    print("-" * 40)
