# LUP: Local Updates Purify for Robust Federated Learning against Poisoning Attacks
- The presented work focuses on addressing cybersecurity threats in Federated Learning (FL), particularly dealing with poisoning attacks and Byzantine clients,
- Our project is developed based on this project https://github.com/JianXu95/SignGuard
## How to run
1) Open the options.py file and customize the options for the dataset, attack, defense method, and other options such as the skew degree of non-IID data. 
2) Run federated_main.py.
3) You will get the training and test performance in a .csv file stored in the same project directory. 
## Requirements
The project requires the following packages to be installed:
- scipy==1.10.1
- seaborn==0.13.0
- torchdata==0.7.1
- torchtext==0.16.1
- torchvision==0.15.2
- torchxrayvision==1.2.1 
* All requirements can be found in the requirements.txt.
## Citation
Please cite our paper (and the respective papers on the methods used) if you use this code in your work:  
 W. Issa, N. Moustafa, B. Turnbull et al., DT-BFL: Digital Twins for Blockchain-enabled Federated Learning in Internet of Things Networks, 
 Ad Hoc Networks (2025),
doi: https://doi.org/10.1016/j.adhoc.2025.103934.




