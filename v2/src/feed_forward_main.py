import argparse
from src.data_loader import load_dataset
from src.model_builder import build_feed_forward_model
from src.trainer import train_model, evaluate_model
from predictor import load_model_and_scalar, predict_new_data
from src.utilis import save_models, result


def main():
    parser = argparse.ArgumentParser(description="Feed Forward Neural Network Trainer")
    parser.add_argument("--mode", choices = ["local", "distributed", "predict"], default="local",
                        help = "Training or prediction mode" )
    parser.add_argument("--data", default="datasets/dataset3_10_11.csv",
                        help = "Path to dataset")
    parser.add_argument("--input_size", type = int, default = 13,
                        help= "Number of input+output columns in dataset")
    parser.add_argument("--epochs", type =int, default=100)
    parser.add_argument("--batch_size", type=int, default= 32)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    #Distributed

    args = parser.parse_args()

    if args.mode ==  "local":
        print("Running local feed forward training..")

        X_train, X_test, y_train,y_test, scaler_X = load_dataset(args.data, args.input_size)
        model = build_feed_forward_model(X_train.shape[1], y_train.shape[1], args.learning_rate)
        history = train_model(model, X_train, y_train, X_test, y_test, args.epochs, args.batch_size)
        test_loss, test_mae = evaluate_model(model, X_test, y_test)
        save_models(model, scaler_X, history, "models")
        print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')
        result(history)



if __name__ == "__main__":
    main()