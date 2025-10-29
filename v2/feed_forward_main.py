import argparse
from src.data_loader import load_dataset, load_csv
from src.model_builder import build_feed_forward_model
from src.trainer import train_model, evaluate_model
from src.predictor import predict_and_analyze
from src.utilis import save_models, result, load_models, ensure_dir
from src.trainer_distributed import train_distributed

#Torch imports
import torch

from src.model_builder_torch import build_feed_forward_model_torch
from src.trainer_torch import train_model_torch, evaluate_model_torch


def main():
    parser = argparse.ArgumentParser(description="Feed Forward Neural Network Trainer")
    parser.add_argument("--mode", choices = ["local", "distributed", "predict","local-torch"], default="local",
                        help = "Training or prediction mode" )
    parser.add_argument("--data", default="datasets/dataset3_10_11.csv",
                        help = "Path to dataset")
    parser.add_argument("--input_size", type = int, default = 13,
                        help= "Number of input+output columns in dataset")
    parser.add_argument("--epochs", type =int, default=100)
    parser.add_argument("--batch_size", type=int, default= 32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--true_data", default="datasets/true_dataset11.csv",
                        help = "Path to true dataset")
    parser.add_argument("--model_path", default="models/model.pkl",
                        help = "Path to the model")
    parser.add_argument("--scaler_path", default="models/scaler.pkl",
                        help = "Path to the scaler_X")

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
    elif args.mode == "predict":
        print("Running prediction and analysis")
        model, scaler_X = load_models(args.model_path, args.scaler_path)
        #true_data = load_csv(args.true_data)
        predict_and_analyze(args.true_data, args.input_size, model, scaler_X)

    elif args.mode == "local-torch":
        print("Running PyTorch FNN ")
        X_train, X_test, y_train,y_test, scaler_X = load_dataset(args.data, args.input_size)
        model, optimizer, criterion = build_feed_forward_model_torch( X_train.shape[1], y_train.shape[1], args.learning_rate)
        history = train_model_torch(model, optimizer, criterion, X_train, y_train, X_test, y_test, epochs=args.epochs, batch_size=args.batch_size)
        evaluate_model_torch(model, X_test, y_test, criterion)
        save_models(model, scaler_X, history, "models")

    elif args.mode == "distributed":
        model, history, scaler_X = train_distributed(
            dataset_path=args.data,
            input_size=args.input_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cluster_hosts=args.cluster,
            rank=args.rank
        )
        ensure_dir("models/distributed")
        save_models(model, scaler_X, history, "models/distributed")


    







if __name__ == "__main__":
    main()