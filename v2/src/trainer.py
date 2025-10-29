def train_model(model,X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    #Train model on single machine and return training history

    history = model.fit(
        X_train, y_train,
        epochs= epochs,
        batch_size = batch_size,
        validation_data=(X_test, y_test),
    )

    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance on the test data
    """

    loss, mae = model.evaluate(X_test, y_test, verbose=2)
    print(f"[Evaluation] Test MSE: {loss: .4f}, MAE: {mae:.4f}")
    return loss, mae
