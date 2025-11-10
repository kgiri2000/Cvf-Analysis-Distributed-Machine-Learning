import argparse
import os
import csv, json
import tensorflow as tf

# from src.data_loader import load_dataset, load_csv
# from src.model_builder import build_feed_forward_model
# from src.trainer import train_model, evaluate_model
from src.predictor import predict_and_analyze
from src.utils  import save_models, result, load_models, ensure_dir
from src.trainer_distributed import train_distributed
from src.trainer_single_gpu import train_single_gpu

#Torch imports
# import torch
# from src.model_builder_torch import build_feed_forward_model_torch
# from src.trainer_torch import train_model_torch, evaluate_model_torch
# sysinfo = {
#     "hostname": platform.node(),
#     "gpu_count": len(tf.config.list_physical_devices('GPU')),
#     "cpu_count": psutil.cpu_count(),
#     "tf_version": tf.__version__,
# }

def main():
    parser = argparse.ArgumentParser(description="Feed Forward Neural Network Trainer")
    parser.add_argument("--mode", choices = ["local-gpu", "distributed", "predict-single","predict-distributed","local-torch"], default="local",
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
    parser.add_argument("--model_path", default="model.pkl",
                        help = "Path to the model")
    parser.add_argument("--scaler_path", default="scaler.pkl",
                        help = "Path to the scaler_X")
    parser.add_argument("--cluster", nargs="+", help = "List of the worker")
    parser.add_argument("--rank", type=int, help = "Worker index")

    #Distributed

    args = parser.parse_args()

    if args.mode ==  "local-gpu":
        print("Running local feed forward training..")

        model, history, scaler_X, total_time_ = train_single_gpu(
            dataset_path=args.data,
            input_size=args.input_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        if history:
            ensure_dir("models/single")
            save_models(model, scaler_X, history, "models/single")

    elif args.mode == "predict-single":
        model_path = os.path.join("models/single",args.model_path)
        scaler_path = os.path.join("models/single",args.scaler_path)
        print(model_path)
        print(scaler_path)
        print("Running prediction and analysis")
        model, scaler_X = load_models(model_path, scaler_path)
        #true_data = load_csv(args.true_data)
        predict_and_analyze(args.true_data, args.input_size, model, scaler_X, args.mode)
    

    elif args.mode == "predict-distributed":
        model_path = os.path.join("models/distributed",args.model_path)
        scaler_path = os.path.join("models/distributed",args.scaler_path)
        print("Running prediction and analysis")
        model, scaler_X = load_models(model_path, scaler_path)
        #true_data = load_csv(args.true_data)
        predict_and_analyze(args.true_data, args.input_size, model, scaler_X, args.mode)

    elif args.mode == "local-torch":
        print("Running PyTorch FNN ")
        X_train, X_test, y_train,y_test, scaler_X = load_dataset(args.data, args.input_size)
        model, optimizer, criterion = build_feed_forward_model_torch( X_train.shape[1], y_train.shape[1], args.learning_rate)
        history = train_model_torch(model, optimizer, criterion, X_train, y_train, X_test, y_test, epochs=args.epochs, batch_size=args.batch_size)
        evaluate_model_torch(model, X_test, y_test, criterion)
        save_models(model, scaler_X, history, "models")

    elif args.mode == "distributed":
        model, history, scaler_X, total_time_ = train_distributed(
            dataset_path=args.data,
            input_size=args.input_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            cluster_hosts=args.cluster,
            rank=args.rank
        )
        if history:
            ensure_dir("models/distributed")
            save_models(model, scaler_X, history, "models/distributed")


    if (args.mode == "local-gpu" or args.mode == "distributed") and  history:
        ensure_dir("results")
        nodes = args.cluster[0].split(" ") if args.cluster else ['single']
        dataset_name = os.path.splitext(os.path.basename(args.data))[0]
        summary_path = os.path.join("results",  f"summary_{args.mode}.csv")
        param_count = model.count_params()
        param_size = param_count * 4/(1024**2) # float32
        metrics= {
            "dataset": os.path.basename(args.data),
            "mode": args.mode,
            "gpu_count": 1 if args.mode == 'local-gpu' else len(nodes),
            "hardware": " 13th Gen Intel(R) Core(TM) i7-13700K & NVIDIA GeForce RTX 4090",
            "input_size": args.input_size,
            "parameters": param_count,
            "train_time": total_time_,
            "train_mae": history["train_mae"][-1],
            "val_mae": history["val_mae"][-1],

        }
        #Append to CSV
        file_exists = os.path.isfile(summary_path)
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames = metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

        





if __name__ == "__main__":
    main()