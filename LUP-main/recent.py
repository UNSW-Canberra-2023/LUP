import numpy as np # linear algebra
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from data_loader import get_dataset
from running import test_classification, benignWorker, byzantineWorker
from models import CNN, ResNet18,CifarCNN
from aggregators import aggregator
from attacks import attack
from options import args_parser
import tools
import time
import copy
from itertools import product
from scipy.stats import skew, kurtosis
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import iqr, median_abs_deviation
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import  mstats
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau


# Temporarily suppress all warnings
warnings.filterwarnings("ignore")

# make sure that there exists CUDA，and show CUDA：
# print(device)
#
# attacks : non, random, noise, signflip, label_flip, byzMean.
#           lie, min_max, min_sum, *** adaptive (know defense) ***
#
# defense : Mean, TrMean, Median, GeoMed, Multi-Krum, Bulyan, DnC, SignGuard.

# set training hype-parameters
# arguments dict

# args = {
#     "epochs": 60,
#     "num_users": 50,
#     "num_byzs": 10,
#     "frac": 1.0,
#     "local_iter": 1,
#     "local_batch_size": 50,
#     "optimizer": 'sgd',
#     "agg_rule": 'SignCheck',
#     "attack": 'non',
#     "lr": 0.2,
#     "dataset": 'cifar',
#     "iid": True,
#     "unbalance": False,
#     "device": device
# }
import numpy as np
from numpy.linalg import svd
from numpy import linalg as LA
from scipy.stats import chi2
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

def clip_gradients_by_norm_and_median(grads_set, factor=1.0):
    # Convert the set of vectors to a tensor
    grads_tensor = torch.stack([torch.tensor(g) for g in grads_set])

    # Calculate the median of gradient norms
    grad_norms = torch.norm(grads_tensor, dim=1)
    median_norm = torch.median(grad_norms)
    
    # Calculate scaling factor to clip gradients
    scaling_factor = factor *  median_norm / (grad_norms + 1e-6)
    scaling_factor = torch.clamp(scaling_factor, max=1.0)
    # Clip gradients element-wise using the scaling factor
    clipped_grads_tensor = grads_tensor * scaling_factor.unsqueeze(1)
    norm_median_ratio = grad_norms / (median_norm + 1e-6)
    # Convert the clipped tensor back to a set of vectors
    clipped_grads_set = [torch.tensor(g.clone().detach() ) for g in clipped_grads_tensor]
    q1 = torch.quantile(norm_median_ratio, 0.25)
    q3 = torch.quantile(norm_median_ratio, 0.75)
    iqr = q3 - q1

    # Define a threshold for filtering based on IQR
    iqr_threshold = 1.5

    # Filter gradients based on IQR
    #filtered_grads_set = [clipped_grads_set[i] for i in range(len(grads_set))
                          #if (q1 - iqr_threshold * iqr) <= grad_norms[i] <= (q3 + iqr_threshold * iqr)]
    #print(len(filtered_grads_set))
    #exit()
    return  clipped_grads_set##clipped_grads_set
# Sample data points

def clip_gradients_by_norm_and_median_2(grads_set, factor=1.0):
    # Convert the set of vectors to a tensor
    grads_tensor = torch.stack(grads_set)

    grad_norm = torch.norm(grads_tensor, dim=1).reshape((-1, 1))
    norm_clip = grad_norm.median(dim=0)[0].item()
    grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
    grads_clip = (grads_tensor/grad_norm)*grad_norm_clipped
    clipped_grads_set = [torch.tensor(g.clone().detach() ) for g in grads_clip]
    #q1 = torch.quantile(norm_median_ratio, 0.25)
    #q3 = torch.quantile(norm_median_ratio, 0.75)
    #iqr = q3 - q1

    # Define a threshold for filtering based on IQR
    #iqr_threshold = 1.5

    # Filter gradients based on IQR
    #filtered_grads_set = [clipped_grads_set[i] for i in range(len(grads_set))
                          #if (q1 - iqr_threshold * iqr) <= grad_norms[i] <= (q3 + iqr_threshold * iqr)]
    #print(len(filtered_grads_set))
    #exit()
    return  clipped_grads_set##clipped_grads_set
# Sample data points
# Sample data points
def cluster_gradient(data):

    # Create linkage matrix using complete linkage method
    linkage_matrix = linkage(data, method='ward')
    #print(data)
    # Get clusters using fcluster
    num_clusters = 2
    clusters = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

    #print("Cluster assignments:", clusters)

    # Map clusters back to the data points
    clustered_data = {}
    
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_data:
            clustered_data[cluster_id] = []
        clustered_data[cluster_id].append(data[i])
    #print(clustered_data)
    # Print data points in each cluster
    #for cluster_id, points in clustered_data.items():
      #  print(f"Cluster {cluster_id}:")
        #for point in points:
          #  print(point)
    cluster_avg_similarities = {}
    for cluster_id in np.unique(clusters):
        indices = np.where(clusters == cluster_id)[0]
        cluster_avg_similarities[cluster_id] = np.mean(data[indices][:, indices])
    cluster_indices = {}
    for cluster_id in np.unique(clusters):
        indices = np.where(clusters == cluster_id)[0]
        cluster_indices[cluster_id] = indices.tolist()
    #print(cluster_indices)
    # Find the cluster with the highest average similarity
    max_avg_similarity_cluster = max(cluster_avg_similarities, key=cluster_avg_similarities.get)
    #print(f"Cluster {max_avg_similarity_cluster} has the highest average similarity.")
    return max_avg_similarity_cluster,cluster_indices
    
    
def get_similarity(g1, g2):
        distance_type="cos"
        if distance_type == "L1":
            return np.sum(np.abs(g1-g2))
        elif distance_type == "L2":
            return np.sum((g1-g2)**2)
        elif distance_type == "cos":
            ng1 = np.sum(g1**2)
            ng2 = np.sum(g2**2)
            if ng1==0.0 or ng2==0.0:
                return 0.0
            else:
                return np.sum(g1 * g2) / np.maximum(np.sqrt(ng1 * ng2), 1e-6)

if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args.dataset)
    # load dataset and user groups
    train_loader, test_loader = get_dataset(args)
    # construct model
    if args.dataset == 'cifar':
        global_model = CifarCNN()
    elif args.dataset == 'fmnist':
        global_model = CNN().to(device)
    else:
        global_model = CNN().to(device)
            
    global_model = global_model#.cuda()

    # optimizer
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                       momentum=0.9, weight_decay=0.0005)
    scheduler = None#MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Training
    # number of iteration per epocj
    iteration = len(train_loader[0].dataset) // (args.local_bs*args.local_iter)
    print(len(train_loader[0].dataset))
    print(args.local_bs*args.local_iter)
    print(iteration)
   
    train_loss, train_acc = [], []
    test_acc = []
    byz_rate = []
    benign_rate = []

    # attack method
    attack_list = ['random', 'signflip', 'noise', 'label_flip', 'lie', 'byzMean', 'min_max', 'min_sum', 'non']
    # attack_id = np.random.randint(9)
    #args.attack = attack_list[attack_id]
    Attack = attack(args.attack)

    # Gradient Aggregation Rule
    GAR = aggregator(args.agg_rule)()
    
    previous_iteration_grads_epoch= []
    previous_iteration_grads_it =[]
    def train_parallel(args, model, train_loader, optimizer, epoch, scheduler):
        print(f'\n---- Global Training Epoch : {epoch+1} ----')
        num_users = args.num_users
        num_byzs = args.num_byzs
        # num_byzs = np.random.randint(1,20)
        device = args.device
        iter_loss = []
        data_loader = []

        for idx in range(num_users):
            data_loader.append(iter(train_loader[idx]))

        previous_iteration_grads = []
        score_matrix_client = np.zeros([args.num_users,1])
        for it in range(iteration):
            #print(it)
            score_matrix_client_it = np.zeros([args.num_users,1])
            m = max(int(args.frac * num_users), 1)
            idx_users = np.random.choice(range(num_users), m, replace=False)
            idx_users = sorted(idx_users)
            local_losses = []
            benign_grads = []
            byz_grads = []
            user_grad_org_all = []
            for idx in idx_users[:num_byzs]:
                grad, loss, user_grad_org= byzantineWorker(model, data_loader[idx], optimizer, args)
                byz_grads.append(grad)
                #user_grad_org_all.append(user_grad_org)
            for idx in idx_users[num_byzs:]:
                grad, loss, user_grad_org = benignWorker(model, data_loader[idx], optimizer, device)
                benign_grads.append(grad)
                local_losses.append(loss)
                user_grad_org_all.append(user_grad_org)


            
            # get byzantine gradient
            model_copy = copy.deepcopy(model)
            byz_grads = Attack(byz_grads, benign_grads, GAR)
            local_grads = benign_grads + byz_grads 
            grads_original = torch.stack(local_grads, dim=0)
            #print(grads_original.shape)
            for byz in byz_grads:
                tools.set_gradient_values(model_copy, byz)
                grad, user_grad_org = tools.get_gradient_values(model_copy)
                user_grad_org_all.append(user_grad_org)
            user_grad_org_test  = [param.flatten()  for param in model.parameters() ]
            #grad_test, user_grad_org_test = tools.get_gradient_values(model)
            #print(len(user_grad_org_all))
            print(len(user_grad_org_all))
            

            grad_scores = []
            #print(user_grad_org_all[0][0].shape)
           # exit()
            num_clients = int(len(user_grad_org_all))
            #print(num_clients)
            sim_matrix = np.zeros((num_clients, num_clients))
            aggregated_grad = [] 
            benign_indices = []
            list_benign_indices =   []
            for layer in range(len(user_grad_org_all[0])):
                all_clients_layer_grad = [ grad[layer].flatten() for grad in user_grad_org_all]

                grads_clip= torch.stack(all_clients_layer_grad, dim=0)
                grads_clip[torch.isnan(grads_clip)] = 0 # remove nan

                # gradient norm-based clustering
                grad_l2norm = torch.norm(grads_clip, dim=1)
                # Step 1: Calculate the median of the population
                median_population = torch.median(grad_l2norm)

                # Step 2: Calculate the deviations from the median
                deviations = torch.abs(grad_l2norm - median_population)

                # Step 3: Find the median of the absolute deviations (scaling factor)
                scaling_factor = torch.median(deviations) 
                sf= scaling_factor  / torch.std(grad_l2norm)
                scaling_factor =   3 * torch.std(grad_l2norm) * sf
                
                # Step 4: Calculate the upper bound
                upper_bound = median_population +  (scaling_factor )

                # Step 5: Calculate the lower bound
                lower_bound = median_population -  (scaling_factor )
                filtered_indices = torch.where((grad_l2norm >= lower_bound) & (grad_l2norm <= upper_bound))
                #ssprint(filtered_indices)

                
                features_list =[]
                layer_grad = []
                list_layer_grad  = []
                if epoch ==0 and it==0 :   
                  user_grad_org_test_layer = user_grad_org_test[layer].flatten()
                elif epoch !=0 and it==0:
                  user_grad_org_test_layer = previous_iteration_grads_epoch[layer].flatten()
                else: 
                  user_grad_org_test_layer = previous_iteration_grads_it[layer].flatten()
                """
                for i,j in product(range(num_clients), range(num_clients)):
                    client_grad_i= user_grad_org_all[i][layer].numpy().flatten()
                    client_grad_j= user_grad_org_all[j][layer].numpy().flatten()
                    sim_matrix[i, j] = get_similarity(client_grad_i,client_grad_j)
                """
                sim_matrix_client = np.zeros((num_clients, 1))
                dev_matrix_client = np.zeros([num_clients,1])
                
                dir_client = np.zeros([num_clients,1])

                user_grad_org_all_filtered = [user_grad_org_all[i] for i in filtered_indices[0].tolist()]
                for i in range(len(user_grad_org_all_filtered)):
                    client_grad_i= user_grad_org_all_filtered[i][layer].numpy().flatten()
                    #client_grad_j= user_grad_org_all[j][layer].numpy().flatten()
                    sim_matrix_client[i, 0] = torch.cosine_similarity(torch.Tensor(client_grad_i),user_grad_org_test_layer, dim=0)
                    dev_matrix_client[i,0] = np.sum(np.abs(client_grad_i - user_grad_org_test_layer.detach().numpy()))
                    dir_client[i,0] = np.dot(client_grad_i , user_grad_org_test_layer.detach().numpy())
                #print(np.mean(sim_matrix_client[0:39,0]))
                #print(np.mean(sim_matrix_client[40:,0]))
                #exit()
                for i in range(len(user_grad_org_all_filtered)):#product(range(num_clients), range(num_clients)):
                    client_grad_i= user_grad_org_all_filtered[i][layer].numpy().flatten()
                    client_grad_i[np.isnan(client_grad_i)] = 0
                    list_layer_grad.append(user_grad_org_all_filtered[i][layer].flatten())
                    vector = client_grad_i
                    replacement_value = 0.0
                    vector[np.isnan(vector)] = replacement_value
                    layer_grad.append(list(client_grad_i))

                    #client_grad_j= user_grad_org_all[j][layer].numpy().flatten()
                    #sim_matrix[i, j] = get_similarity(client_grad_i,client_grad_j)
                    #client_grad_flatten= user_grad_org_all[client][layer].numpy().flatten()
                    #print(client_grad_flatten.shape)
                    positive_count = np.sum(vector > 0)
                    negative_count = np.sum(vector < 0)
                    zero_count = np.sum(vector == 0)

                    # Basic Statistical Features
                    mean_value = np.mean(vector)
                    median_value = np.median(vector)
                    std_deviation = np.std(vector)
                    skewness = skew(vector)
                    kurt = kurtosis(vector)

                    # Additional Features
                    sum_value = np.sum(vector)
                    min_value = np.min(vector)
                    max_value = np.max(vector)
                    range_value = max_value - min_value
                    variance = np.var(vector)
                    from scipy.stats import circmean, circvar
                    from scipy.stats import entropy
                    # Robust Features
                    mad = np.median(np.abs(vector - median_value))
                    iqr = np.percentile(vector, 75) - np.percentile(vector, 25)
                    #print(autocorr)
                    #mad = median_absolute_deviation(vector)
                    sim = np.mean(sim_matrix[i,:])
                    #winsorized_mean = mstats.winsorize(vector, limits=0.1).mean()
                    #mean_direction = circmean(vector)
                    #circular_variance = circvar(vector)
                    #if epoch >2:
                    absolute_deviation = np.sum(np.abs(vector - user_grad_org_test_layer.detach().numpy()))
                    dir = np.dot(vector , user_grad_org_test_layer.detach().numpy())
                    grad_l2norm = np.linalg.norm(vector)
                    score = np.abs(grad_l2norm -median_population ) / scaling_factor
                    #else:
                    #    absolute_deviation=0.0
                    #pearson_corr, _ = pearsonr(vector, user_grad_org_test_layer)
                    #spearman_corr, _ = spearmanr(vector, user_grad_org_test_layer)
                    #kendall_tau, _ = kendalltau(vector, user_grad_org_test_layer)
                    # np.mean(sim_matrix[i,:])
                    # Store features for the current vector in a list   score_matrix_client[i,0]s
                    vector_features = [
                        positive_count, negative_count, zero_count,kurt,skewness
                               ,absolute_deviation,mad,median_value,iqr,score,score_matrix_client[i,0]
                       
                    ]
                 
                    #print(vector_features)
                    tensor_data = torch.tensor(vector_features)
                    replacement_value = 0.0
                    tensor_data[torch.isnan(tensor_data)] = replacement_value
                    vector_features = tensor_data.tolist()

                    #print(vector_features)
                    #replacement_value = 0.0
                    #tensor_data[torch.isnan(tensor_data)] = replacement_value
                    features_list.append(vector_features)



                # Standardize the feature vectors
                scaler = MinMaxScaler()
                standardized_features = scaler.fit_transform(features_list)
               # print(features_list)
                # Perform PCA for dimensionality reduction (optional)
                #pca = PCA(n_components=2)
                reduced_features = np.array(standardized_features)
               # print(reduced_features.shape)
               
                # Specify the number of clusters
                num_clusters = 2

                # Create Agglomerative Clustering model
                agg_clustering = AgglomerativeClustering()

                # Fit the model and predict cluster labels
                cluster_labels = agg_clustering.fit_predict(reduced_features)
                from collections import Counter

                #print(set(cluster_labels))
                value_indices = {value: [index for index, val in enumerate(cluster_labels) if val == value] for value in set(cluster_labels)}
                #print(value_indices)
                #element_counts = Counter(cluster_labels)
                #print(element_counts)
                
                # Print the counts
                #for element, count in element_counts.items():
                 #   print(f"Element {element} appears {count} times")
                # Plot the dendrogram
                #from scipy.cluster.hierarchy import linkage, dendrogram
                #linkage_matrix = linkage(standardized_features, method='ward', metric='euclidean')
                #dendrogram(linkage_matrix)
                #plt.xlabel('Sample Index')
                #plt.ylabel('Distance')
                #plt.title('Dendrogram')
                #plt.show()

                # Visualize the clusters (for 2D data)
                #plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')
                #plt.xlabel('Principal Component 1')
                #plt.ylabel('Principal Component 2')
                #plt.title('Hierarchical Clustering')
                #plt.show()

                # Print the cluster labels
                #print("Cluster Labels:", cluster_labels)
                import torch.nn.functional as F
                

                #list_layer_grad = clip_gradients_by_norm_and_median_2(list_layer_grad)
                layer_selected_grad = torch.stack(list_layer_grad)

                cluster_1_layer =list(torch.index_select(layer_selected_grad, 0, torch.tensor(value_indices[0])))
                cluster_2_layer = list(torch.index_select(layer_selected_grad, 0, torch.tensor(value_indices[1])))
                layer_selected_grad_1 = torch.mean(torch.stack(cluster_1_layer),dim=0)
                layer_selected_grad_2 = torch.mean(torch.stack(cluster_2_layer),dim=0)

                absolute_deviation_1_c = np.mean(np.abs(layer_selected_grad_1.detach().numpy() - user_grad_org_test_layer.detach().numpy()))
                absolute_deviation_2_c= np.mean(np.abs(layer_selected_grad_2.detach().numpy() - user_grad_org_test_layer.detach().numpy()))


                correlation1_var =  torch.var(layer_selected_grad_1, dim=0)
                correlation2_var =  torch.var(layer_selected_grad_2, dim=0)
                correlation1_var =torch.mean(correlation1_var).item()
                correlation2_var =torch.mean(correlation2_var).item()
               
                #print(layer_selected_grad)
                #avg_layer_grad =   torch.sum(torch.stack(layer_selected_grad_1,dim=0), dim=0) / len(layer_selected_grad_1) 
                #avg_layer_grad_other=   torch.sum(torch.stack(layer_selected_grad_2,dim=0), dim=0) / len(layer_selected_grad_2) 
               # user_grad_org_test_layer = user_grad_org_test[layer].flatten()
               # correlation1= torch.cosine_similarity(avg_layer_grad, user_grad_org_test_layer,dim=0).item()
               # correlation2 =  torch.cosine_similarity(avg_layer_grad_other, user_grad_org_test_layer,dim=0).item()
                """
                correlation1 =  torch.cosine_similarity(layer_selected_grad_1,user_grad_org_test_layer, dim=0)
                correlation2 =  torch.cosine_similarity(layer_selected_grad_2,user_grad_org_test_layer, dim=0)
                absolute_deviation_1 = np.mean(np.abs(layer_selected_grad_1.detach().numpy() - user_grad_org_test_layer.detach().numpy()))
                absolute_deviation_2 = np.mean(np.abs(layer_selected_grad_2.detach().numpy() - user_grad_org_test_layer.detach().numpy()))
                correlation1_var =  torch.var(layer_selected_grad_1, dim=0)
                correlation2_var =  torch.var(layer_selected_grad_2, dim=0)
                correlation1_var =torch.mean(correlation1_var).item()
                correlation2_var =torch.mean(correlation2_var).item()
                other_cluster_indices = value_indices[0]
                C
                selected_list = cluster_1_layer
                from scipy.special import kl_div
                from scipy.spatial import distance
                from scipy.special import rel_entr
                """
                sim_cluster_1 = np.mean(sim_matrix_client[value_indices[0]])
                sim_cluster_2 = np.mean(sim_matrix_client[value_indices[1]])
                score_cluster_1 = np.sum(score_matrix_client[value_indices[0]])
                score_cluster_2 = np.sum(score_matrix_client[value_indices[1]])
                absolute_deviation_1 = np.mean(dev_matrix_client[value_indices[0]])  #np.mean(np.abs(layer_selected_grad_1.detach().numpy() - user_grad_org_test_layer.detach().numpy()))
                absolute_deviation_2 = np.mean(dev_matrix_client[value_indices[1]])   #np.mean(np.abs(layer_selected_grad_2.detach().numpy() - user_grad_org_test_layer.detach().numpy()))
                dot_1 = np.sum(dir_client[value_indices[0]] > 0 )
                dot_2 = np.sum(dir_client[value_indices[1]] > 0 )
                
                selected_list = cluster_1_layer
                other_cluster_indices = value_indices[0]
                #print(str(correlation2_var)+"---"+str(correlation1_var))
                grad_l2norm_1 = torch.norm(torch.stack(cluster_1_layer, dim=0),dim=1).numpy()
                norm_med_1 = np.mean(grad_l2norm_1)
                grad_l2norm_2 = torch.norm(torch.stack(cluster_2_layer, dim=0),dim=1).numpy()
                norm_med_2 = np.median(grad_l2norm_2)
                index=0
                if score_cluster_2 > score_cluster_1  and absolute_deviation_2 < absolute_deviation_1 : #(correlation2 > correlation1) and (correlation2_var < correlation1_var) and (absolute_deviation_2 < absolute_deviation_1):
                     other_cluster_indices = value_indices[1]
                     selected_list = cluster_2_layer
                     index =1

                #benign_indices =benign_indices + other_cluster_indices
                #print("selected cluster" +str(index))
                score_matrix_client [other_cluster_indices,0] += 1
                #score_matrix_client [value_indices[index],0] -= 1
                #print(score_matrix_client)
                #print(other_cluster_indices)
                """  
                client_index =0 
                for selected_client in selected_list: 
                    flatten_client_grad = selected_client
                    dot = torch.dot(user_grad_org_test_layer, flatten_client_grad.flatten())
                    if dot < 0 : 
                        term_2 = torch.div(torch.multiply(user_grad_org_test_layer ,dot ),(torch.sum(user_grad_org_test_layer ** 2)))
                        flatten_client_grad_projected = torch.sub(flatten_client_grad.flatten(), term_2)
                        selected_list[client_index]  = flatten_client_grad_projected
                    client_index +=1
                """      
                """  
                batch_norm_layer = nn.BatchNorm1d(selected_list[0].shape[0])
                selected_list = batch_norm_layer(torch.stack(selected_list,dim=0))
                selected_list = selected_list.tolist()
                selected_list = [torch.tensor(sublist) for sublist in selected_list]
                """ 
                #selected_list = clip_gradients_by_norm_and_median_2(selected_list)
                print(len(selected_list))
                grads_clip= torch.stack(selected_list, dim=0)
                grads_clip[torch.isnan(grads_clip)] = 0 # remove nan

                # gradient norm-based clustering
                grad_l2norm = torch.norm(grads_clip, dim=1)
                # Step 1: Calculate the median of the population
                median_population = torch.median(grad_l2norm)

                # Step 2: Calculate the deviations from the median
                deviations = torch.abs(grad_l2norm - median_population)

                # Step 3: Find the median of the absolute deviations (scaling factor)
                scaling_factor = torch.median(deviations) 
                #sf= scaling_factor  / torch.std(grad_l2norm)
                #sscaling_factor =   3 * torch.std(grad_l2norm) * sf
                
                # Step 4: Calculate the upper bound
                upper_bound = median_population +  (scaling_factor )

                # Step 5: Calculate the lower bound
                lower_bound = median_population -  (scaling_factor )
                filtered_indices = torch.where((grad_l2norm >= lower_bound) & (grad_l2norm <= upper_bound))
                #print(filtered_indices)
                layer_selected_grad_projected =selected_list#clip_gradients_by_norm_and_median_2(selected_list)
                benign_indices =benign_indices +  [other_cluster_indices[i] for i in filtered_indices[0].tolist()]
                list_benign_indices.append(other_cluster_indices)
                #print(list_benign_indices)
                   

                
                """
                #print(sim_matrix)    
                max_avg_similarity_cluster,cluster_indices = cluster_gradient(sim_matrix)
                other_cluster = 1
                if max_avg_similarity_cluster ==1:
                        other_cluster = 2
                other_cluster_indices = cluster_indices[0]

                cluster_indices = cluster_indices[max_avg_similarity_cluster]
                #print(cluster_indices)
                layer_selected_grad = []
                
                for selected_client in cluster_indices: 
                    layer_selected_grad.append(user_grad_org_all[selected_client][layer])
               # print(layer_selected_grad[1].shape)
                avg_layer_grad = sum(layer_selected_grad) / len(layer_selected_grad)
                #print(avg_layer_grad)
                
                #print(aggregated_grad)
                #exit()  
                """
                """
                layer_selected_grad_projected = []
                #layer_selected_grad_projected = layer_selected_grad_projected + layer_selected_grad
                for selected_client in other_cluster_indices: 
                    flatten_client_grad = user_grad_org_all[selected_client][layer]
                    #print(type(flatten_client_grad))
                    #exit()
                    #dot = torch.dot(avg_layer_grad.flatten(), flatten_client_grad.flatten()) 
                    
                    
                   # if dot < 0 : 
                       # term_2 = torch.div(torch.multiply(avg_layer_grad ,dot ),(torch.sum(avg_layer_grad ** 2)))
                        #flatten_client_grad_projected = torch.sub(flatten_client_grad , term_2)
                   # else:
                    flatten_client_grad_projected = flatten_client_grad.flatten()

                    layer_selected_grad_projected.append(flatten_client_grad_projected)
                 """   
                #avg_layer_grad_projected = torch.sum(torch.stack(layer_selected_grad_projected,dim=0), dim=0) / len(layer_selected_grad_projected)
                #previous_iteration_grads.append(avg_layer_grad_projected)
                #aggregated_grad =  list(aggregated_grad)+ list(avg_layer_grad_projected) 
                #print( len())
               # print(layer_selected_grad[1].shape)
                #exit()
                """
                avg_layer_grad_projected = torch.sum(torch.stack(layer_selected_grad_projected,dim=0), dim=0) / len(layer_selected_grad_projected)
                def clip_gradients_by_median(grads, factor=1.0):
                        # Calculate the median of gradient norms
                        grad_norms = torch.norm(grads, dim=1)
                        median_norm = torch.median(grad_norms)
                        
                        # Calculate scaling factor to clip gradients
                        scaling_factor = factor * median_norm / (grad_norms + 1e-6)
                        
                        # Clip gradients element-wise using the scaling factor
                        clipped_grads = grads * scaling_factor.unsqueeze(1)
                        
                        return clipped_grads
                #aggregated_grad =  list(aggregated_grad)+ list(avg_layer_grad_projected) 
                #aggregated_grad = list(sum(aggregated_grad) / len(aggregated_grad))
                #print(avg_layer_grad_projected.shape)
                if len(avg_layer_grad_projected.shape) ==1:
                    avg_layer_grad_projected = avg_layer_grad_projected.view(-1, 1)
                grad_norm = clip_gradients_by_median(avg_layer_grad_projected)
                #print(grad_norm.shape)
                
                #norm_clip = grad_norm.reshape((-1, 1)).median(dim=0)[0].item()
                #print(norm_clip)
                #print(grad_norm.shape)
                
                #grad_norm_clipped = torch.clamp(grad_norm, 1, norm_clip)
                
                #print(grad_norm_clipped.shape)
                
                #print(avg_layer_grad_projected.reshape((-1, 1)).shape)
                #exit()
                #print(grad_norm_clipped.shape)
                #print(avg_layer_grad_projected.shape)
                # Reshape avg_layer_grad_projected to a 2D tensor
                #avg_layer_grad_projected_reshaped = avg_layer_grad_projected.clone().detach()
                #avg_layer_grad_projected_reshaped = avg_layer_grad_projected_reshaped.reshape((-1, 1))
                #print(avg_layer_grad_projected_reshaped.shape)
                #print(avg_layer_grad_projected.shape)
                #grad_norm_clipped= grad_norm_clipped.reshape((-1, 1))
                #print(grad_norm_clipped.shape)
                #grads_clip = (avg_layer_grad_projected/grad_norm) *(grad_norm_clipped)
                #print(grads_clip.shape)
               # exit()
               """
                #aggregated_grad =  aggregated_grad + list(grad_norm.flatten()) 
                #print(aggregated_grad)
                #exit()
                #exit()   
                               
            # get global gradient
            #print(len(aggregated_grad))
            #exit()
            #global_grad, selected_idx, isbyz = GAR.aggregate(local_grads, f=num_byzs, epoch=epoch, g0='grad_0', iteration=it)
            #byz_rate.append(isbyz)
           # benign_rate.append((len(selected_idx)-isbyz*num_byzs)/(num_users-num_byzs))
            # update global model

            #aggregated_grad=torch.stack(aggregated_grad, dim=0)
            #print("------------------------------")
            #print(list(set(benign_indices)))
            grad_norm = torch.norm(grads_original, dim=1).reshape((-1, 1))
            norm_clip = grad_norm.median(dim=0)[0].item()
            grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
            grads_clip = (grads_original/grad_norm)*grad_norm_clipped
            intersection_set = set(list_benign_indices[0]).intersection(*list_benign_indices[1:])

            print(intersection_set)
            if len(intersection_set) !=0:
                global_grad = grads_original[list(set(intersection_set))].mean(dim=0)
            else:
                _,previous_iteration_grads_it =  tools.get_gradient_values(model)
            #global_grad = grads_original[list(set(benign_indices))].mean(dim=0)
            #print(global_grad.shape)
            tools.set_gradient_values(model, global_grad)
            optimizer.step()
            _,previous_iteration_grads_it =  tools.get_gradient_values(model)
            loss_avg = sum(local_losses) / len(local_losses)
            iter_loss.append(loss_avg)

            if (it+1) % 1 == 0:  # print every 10 local iterations
                print('[epoch %d, %.2f%%] loss: %.5f' %
                      (epoch + 1, 100 * ((it + 1)/iteration), loss_avg), "--- byz. attack succ. rate:", 0, '--- selected number:', 0)

        if scheduler is not None:
            scheduler.step()

        return iter_loss,previous_iteration_grads_it

    for epoch in range(args.epochs):
        loss, previous_iteration_grads_epoch= train_parallel(args, global_model, train_loader, optimizer, epoch, scheduler)
        acc = test_classification(device, global_model, test_loader)
        print("Test Accuracy: {}%".format(acc))