import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from data_loader import get_dataset
from running import test_classification, benignWorker, byzantineWorker
from models import CNN, ResNet18, CifarCNN, RNNClassifier, MLP
from aggregators import aggregator
from attacks import attack
from options import args_parser
import tools
import time
import copy
import warnings
import os
import pandas as pd
import time

# Set random seed for PyTorch CPU
torch.manual_seed(0)
# Temporarily suppress all warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__': #'label_flip', 'sign_flip','lie','byzMean', 'min_max',
    attack_keys = [ 'byzMean']#['lie', 'byzMean', 'min_max', 'min_sum', 'non', 'mpfa']  #'label_flip', 'sign_flip','random', 'noise', 'label_flip', 


  #'label_flip','lie', 'random','byzMean',  'noise', 'label_flip', 
               #'min_sum'   attacks_keys = 
#['fang']#'label_flip', 'min_sum','lie','noise','sign_flip','byzMean',  'min_max', 'non'] 'label_flip','fang','mpfa','min_max','random','byzMean', 
  #'Bulyan','SignGuard','DnC',   
    """
    ,
    'SignGuard','DnC', 
     
    'Mean',
    'TrMean',
    'Median',
    
    'GeoMed',
    'Multi-Krum',
   
    'clipcluster',
    'centerclip'
  'Multi-Krum',
     """
    GAR_keys =[ 'LUP']  
    #'SignGuard', 'Median', 'GeoMed', 'Multi-Krum', 'DnC', 'SignGuard', 'clipcluster'



   
    for attack_name in attack_keys:
        for gar_name in GAR_keys:
            args = args_parser()
            device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
            print(device)
            print(args.dataset)

            # Load dataset and user groups
            train_loader, test_loader = get_dataset(args)
            
            # Construct model
            if args.dataset == 'ton_iot':
                global_model = MLP()  # ResNet18()
            elif args.dataset == 'AGNews':
                global_model = RNNClassifier()  # ResNet18()
            elif args.dataset == 'cifar':
                global_model =  CifarCNN()
            elif args.dataset == 'fmnist':
                global_model = CNN().to(device)
            else:
                global_model = CNN().to(device)

            

            # Verify that parameters are initialized with zeros
            # for param in global_model.parameters():
            #    print(param)

            # momentum=0.9,
            # Optimizer
                        # print(score_matrix_client)
            
            #weight_decay=0.0005 # momentum=0.9
            scheduler = None  # MultiStepLR(optimizer, milestones=[100], gamma=0.1)
            criterion = nn.CrossEntropyLoss()
            # Training
            # Number of iteration per epoch
            iteration = len(train_loader[0].dataset) // (args.local_bs * args.local_iter)
            print(len(train_loader[0].dataset))
            print(args.local_bs * args.local_iter)
            print(iteration)

            train_loss, train_acc = [], []
            test_acc = []
            byz_rate = []
            benign_rate = []

            # Attack method
            attack_list = ['noise', 'label_flip', 'lie', 'byzMean', 'min_max', 'min_sum', 'non', 'random', 'signflip']
            # attack_id = np.random.randint(9)
            # args.attack = attack_list[attack_id]
            Attack = attack(attack_name)

            # Gradient Aggregation Rule
            GAR = aggregator(gar_name)()

            previous_iteration_grads_epoch = []
            previous_iteration_grads_it = []
            score_matrix_client = np.zeros([args.num_users, 1])
            model_copy = copy.deepcopy(global_model)
            def train_parallel(args, train_loader, global_model,epoch, scheduler):
                #model = global_model
                print(f'\n---- Global Training Epoch : {epoch+1} ----')
                num_users = args.num_users
                num_byzs = args.num_byzs
                # num_byzs = np.random.randint(1,20)
                device = args.device
                iter_loss = []
                data_loader = []
                
                previous_iteration_grads = []


                
                if True:

                    model_copies = []
                    opt = []
                    for idx in range(num_users):
                        #data_loader.append(iter(train_loader[idx]))
                        local_model = copy.deepcopy(global_model)
                        local_model.load_state_dict(global_model.state_dict())
                        model_copies.append(local_model)
                        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr)#,momentum=0.99,weight_decay=0.0005)
                        opt.append(optimizer)

                    m = max(int(args.frac * num_users), 1)
                    idx_users = np.random.choice(range(num_users), m, replace=False)
                    idx_users = sorted(idx_users)
                    local_losses = []
                    benign_grads = []
                    byz_grads = []
                    local_models = []
                    user_grad_org_all = []
                    for idx in idx_users[:num_byzs]:
                        grad, loss, user_grad_org,model = byzantineWorker(
                            model_copies[idx],opt[idx], train_loader[idx], args,idx)
                        byz_grads.append(grad.clone().detach())
                        user_grad_org_all.append(user_grad_org)
                        local_models.append(model)
                    for idx in idx_users[num_byzs:]:
                        #x,y=tools.get_gradient_values(model_copies[idx])
                        #print(x)
                        grad, loss, user_grad_org,model = benignWorker(
                            model_copies[idx],opt[idx], train_loader[idx], device,args,idx)
                        benign_grads.append(grad.clone().detach())
                        local_losses.append(loss)
                        user_grad_org_all.append(user_grad_org)
                        local_models.append(model)
                
                    # Get byzantine gradient
                    # user_grad_org_test  = [param.detach().flatten()  for param in model.parameters() ]
                    user_grad_org_test = torch.zeros_like(benign_grads[0])

                    byz_grads = Attack(byz_grads, benign_grads, GAR)
                    local_grads = benign_grads + byz_grads
                    # grads_original = torch.stack(local_grads, dim=0)
                    # print(grads_original.shape)
                    #for byz in byz_grads:
                        #tools.set_gradient_values(model_copy, byz)
                        #grad, user_grad_org = tools.get_gradient_values(model_copy)
                        #user_grad_org_all.append(user_grad_org)
                    #print(local_grads)
                    previous_grad = []
                    if epoch == 0 :
                        previous_grad = user_grad_org_test
                    else :
                        previous_grad = previous_iteration_grads_epoch
                    
                    # Get global gradient
                    global_grad = []
                    if gar_name != 'LUP':
                        start_time = time.time()
                        global_grad, selected_idx, isbyz = GAR.aggregate(
                            local_grads, f=num_byzs, epoch=epoch, g0='grad_0', iteration=0)
                        end_time = time.time()

                        # Calculate running time
                        running_time = end_time - start_time

                        print("Running time:", running_time, "seconds")
                    else:
                        start_time = time.time()
                        global_grad, selected_idx, isbyz = GAR.aggregate(
                            local_grads, user_grad_org_all, previous_grad, score_matrix_client, f=num_byzs, epoch=epoch, g0='grad_0', iteration=0)
                        end_time = time.time()

                        # Calculate running time
                        running_time = end_time - start_time

                        print("Running time:", running_time, "seconds")
                    byz_rate.append(isbyz)
                    benign_rate.append(
                        (len(selected_idx)-isbyz*num_byzs)/(num_users-num_byzs))
                    # Update global model
                    global_model.load_state_dict(local_models[0].state_dict())
                    tools.set_gradient_values(global_model, global_grad)
                    acc = test_classification(
                    epoch, device, global_model, test_loader, criterion, output_csv)
                    print("Test Accuracy: {}%".format(acc))
                    #print(global_grad)
                    #optimizer.step()

                    previous_iteration_grads_it, _ = tools.get_gradient_values(global_model)

                    loss_avg = sum(local_losses) / len(local_losses)
                    iter_loss.append(loss_avg)
                    output_csv_indices = "results_index_"+args.dataset+"_" + \
                    attack_name +"_"+str(args.num_byzs)+"_"+gar_name+"_"+str(args.skew)+".csv"
                    if os.path.exists(output_csv_indices) and epoch == 0 :
                        # Delete the file
                        os.remove(output_csv_indices)
                    df = pd.DataFrame(selected_idx, columns=['Index'])
                    df.to_csv(output_csv_indices, index=False,mode='a')


                    if (epoch + 1) % 1 == 0:  # print every 10 local iterations
                        print('[epoch %d, %.2f%%] loss: %.5f' %
                            (epoch + 1, 100 * ((epoch + 1)/args.epochs), loss_avg), "--- byz. attack succ. rate:", isbyz, '--- selected number:', len(selected_idx))

                if scheduler is not None:
                    scheduler.step()
                return iter_loss, previous_iteration_grads_it,  running_time

            for epoch in range(args.epochs):
                output_csv = "results_"+args.dataset+"_" + \
                    attack_name+"_"+str(args.num_byzs)+"_"+gar_name+"_"+str(args.skew)+".csv"
                output_csv_indices = "results_index_"+args.dataset+"_" + \
                    attack_name+"_"+str(args.num_byzs)+"_"+gar_name+"_"+str(args.skew)+".csv"
                if os.path.exists(output_csv) and epoch == 0:
                    # Delete the file
                    os.remove(output_csv)
                loss, previous_iteration_grads_epoch ,  running_time= train_parallel(
                    args, train_loader,global_model, epoch, scheduler)
                
                acc = test_classification(
                    epoch, device, global_model, test_loader, criterion, running_time,output_csv)
                print("Test Accuracy: {}%".format(acc))
