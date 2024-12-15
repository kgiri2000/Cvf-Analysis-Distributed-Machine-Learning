import os
from src.dijkstra import Dijkstra
from src.dataset import save_dataset
#Can create the whole dataset from  start node to max node
#Also can create the dataset for specific node number

def generatedataset(*args):

    if len(args) < 0 or len(args) > 3:
        return 0
    elif len(args) == 3:
        start_node = args[0]
        max_node = args[1]
        max_predict_node = args[2]
        file_path = f"datasets/dataset{start_node}_{max_node}_{max_predict_node}.csv"
        if os.path.exists(file_path):
            os.remove(file_path)
            
        #Generate whole dataset
        all_graphs = [f"graphs/dijkstra_{i}.txt" for i in range(start_node, max_node+1)]
        for graph_path in all_graphs:
            result_path = "datasets"
            cvf  = Dijkstra(graph_path, result_path, max_predict_node)
            cvf.analyse()
            #Save the data each time with open mode
            res = save_dataset(cvf.dataset, file_path, append= True)
    elif len(args) == 2:
        #Generate true dataset
        print("Number of args: ", len(args))
        node_num = args[0]
        max_pred = args[1]
        file_path = f"datasets/true_dataset{node_num}.csv"
        graph_path = f"graphs/dijkstra_{node_num}.txt"
        result_path = "datasets"
        cvf = Dijkstra(graph_path, result_path, max_pred)
        cvf.analyse()
        res = save_dataset(cvf.dataset, file_path,append=True )
    else:
        print("Incorrect Input")
        return 0
        




        
if __name__ == "__main__":
    generatedataset(3,10,11)
    generatedataset(11)
    #Generating dataset for graph node = 10 for prediction
    print("Generating Dataset")
    generatedataset(10,11)

                                                                                                             

    

