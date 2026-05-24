import pandas as pd
from eda import preprocess_data
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_linear_regression_pytorch(
    X_train,
    y_train,
    X_val,
    y_val,
    y_scaler,
    epochs=5000,
    learning_rate=0.01
):

    # ==========================================
    # 1. Convert sang tensor
    # ==========================================

    X_train_tensor = torch.tensor(
        X_train.values,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val.values,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val.values,
        dtype=torch.float32
    )

    # ==========================================
    # 2. Define model
    # ==========================================

    input_size = X_train.shape[1]

    # vô inpuet_size, ra output 1 giá trị dự đoán
    model = nn.Linear(input_size, 1)

    # ==========================================
    # 3. Loss function
    # ==========================================

    criterion = nn.MSELoss()

    # ==========================================
    # 4. Optimizer
    # ==========================================

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    # Early Stopping setup
    best_val_loss = float('inf')
    counter = 0
    patience = 50  # dừng nếu validation loss không cải thiện trong 50 epochs

    # Loss tracking
    train_losses = []
    val_losses = []

    # ==========================================
    # 5. Training loop
    # ==========================================

    for epoch in range(epochs):

        # Forward
        # Dự đoán trên tập train
        predictions = model(X_train_tensor)

        # Tính loss giữa dự đoán và giá trị thật trên tập train
        loss = criterion(
            predictions,
            y_train_tensor
        )

        # Backward

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Validation
        if (epoch + 1):
            # Để không lưu grads của validation, vì không cần thiết
            with torch.no_grad():

                val_predictions = model(X_val_tensor)

                val_loss = criterion(
                    val_predictions,
                    y_val_tensor
                )
            # ==========================================
            # Early Stopping
            # ==========================================

            if val_loss.item() < best_val_loss:

                best_val_loss = val_loss.item()

                counter = 0

            else:

                counter += 1

            if counter >= patience:

                print(f"\nEarly stopping at epoch {epoch+1}")

                break
            
            # Lưu loss vào danh sách
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {loss.item():.6f} "
                f"Validation Loss: {val_loss.item():.6f}"
            )

    # ==========================================
    # 6. Predict validation
    # ==========================================

    # Dự đoán với model mới được train xong
    with torch.no_grad():

        y_pred_scaled = model(X_val_tensor)

    # ==========================================
    # 7. Convert prediction về giá thật
    # ==========================================

    y_pred_real = y_scaler.inverse_transform(
        y_pred_scaled.numpy()
    )

    y_val_real = y_scaler.inverse_transform(
        y_val.values
    )

    print("\n=== Real Predictions ===")

    for i in range(y_pred_real.shape[0]):

        print(
            f"Predicted: {y_pred_real[i][0]:,.2f} | "
            f"Actual: {y_val_real[i][0]:,.2f}"
        )

    # ==========================================
    # 8. Plot loss
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('result/loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==========================================
    # 9. Plot predictions vs actual (linear function)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Scatter plot: Actual vs Predicted
    plt.scatter(y_val_real, y_pred_real, alpha=0.6, s=50, label='Predictions')
    
    # Line: Đường thẳng lý tưởng (y = x)
    min_val = min(y_val_real.min(), y_pred_real.min())
    max_val = max(y_val_real.max(), y_pred_real.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Perfect Prediction (y=x)')
    
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Linear Regression: Predicted vs Actual')
    plt.legend()
    plt.grid(True)
    plt.savefig('result/predictions_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model

if __name__ == "__main__":
    data = pd.read_csv("23_HOMES.csv")
    X_train_scaled,X_val_scaled,y_train_scaled, y_val_scaled,x_scaler,y_scaler = preprocess_data(data)
    model = train_linear_regression_pytorch(
    X_train_scaled,
    y_train_scaled,
    X_val_scaled,
    y_val_scaled,
    y_scaler
)