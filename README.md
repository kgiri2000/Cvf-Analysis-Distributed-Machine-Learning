# Cvf-Analysis-Distributed-Machine-Learning

How to run the code?

pip install -r requirements.txt

# Making dataset:
Default:
### Dataset(node n = 3 to node n = 10)
### Dataset for actual node n = 11(For prediction)
### Dataset for actual node n = 10( For predictions)

>python generatedataset.py

# Training FF network
Default:
>python train_ff_model.py

Model will be saved under models

# Prediction using FF
#For node number n = 11
Default:
For node n = 11
Change the value of true_dataset in main.py
true_dataset = "datasets/true_dataset11.csv"

>python main.py

comparison plot will be generated inside plots as comparison_counts.png with evaluation plot as well as predicted Ar  will be saved inside the datasets as predict_dataset11_using11.csv as well as predicted Ar count will be as predicted_Ar_counts11_using11.csv

For node number n = 10
Change the value of true_dataset in main.py
true_dataset = "datasets/true_dataset10.csv"

>python main.py

comparison plot will be generated inside plots as comparison_counts.png with evaluation plot as well as predicted Ar  will be saved inside the datasets as predict_dataset10_using10.csv as well as predicted Ar count will be as predicted_Ar_counts10_using10.csv

# Training using Distributed Machine Learning
Default:
>python train_dml_model.py

Model will be save under models

# Prediction using Distributed Machine Learning
Similar to FNN
