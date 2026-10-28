import pandas as pd
import sys

def validate():
    try:
        df = pd.read_csv("metrics.csv")
    except Exception as e:
        print("FAIL: metrics.csv introuvable.")
        sys.exit(1)

    if len(df) == 0:
        print("FAIL: metrics.csv est vide.")
        sys.exit(1)

    last_row = df.iloc[-1]
    accuracy = last_row["accuracy"]
    loss = last_row["loss"]

    # Validation criteria
    # Accuracy >= 0.85 (Non-IID acceptable)
    # Loss <= 0.15
    
    passed = True
    if accuracy < 0.85:
        print(f"FAIL: Accuracy {accuracy:.4f} < 0.85")
        passed = False
    
    if loss > 0.15:
        print(f"FAIL: Loss {loss:.4f} > 0.15")
        passed = False
        
    if passed:
        print(f"PASS: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    validate()
