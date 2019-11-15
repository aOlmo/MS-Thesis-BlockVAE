import itertools
import os

CONFIGURATION_BASE_DIR = "configurations"
RESULTS_BASE_DIR = "configurations/results"
STATIC_BASE_STRING_LFW = """
    red_only = 0

    num_layers = 1
    vae_loss_type = sad
    conditional = 0
    epsilon_std = 1.0
    training_mean = 0
    lr = 0.001
     
    block_vae_weights = weights_block_vae.h5
    block_cnn_weights = weights_block_cnn.h5
     
    num_samples = 10000
    batch_size = 100
    epochs = 300
     
    block_cnn_nb_epoch = 30
    block_cnn_batch_size = 100
"""

STATIC_BASE_STRING_CIFAR10 = """
    dataset = cifar10
    red_only = 0
    
    num_layers = 1
    vae_loss_type = sad
    conditional = 0
    epsilon_std = 1.0
    training_mean = 0
    lr = 0.001
     
    block_vae_weights = weights_block_vae.h5
    block_cnn_weights = weights_block_cnn.h5
     
    num_samples = 10000
    batch_size = 100
    epochs = 800
     
    block_cnn_nb_epoch = 50
    block_cnn_batch_size = 100
"""

STATIC_BASE_STRING = """
    categorical = 0
    num_layers = 1
    
    num_samples = 10000
    batch_size = 100
    conditional = 0
    epochs = 150
    red_only = 0
    vae_loss_type = binary
    
    epsilon_std = 0.01
    training_mean = 0
    
    # Only for BlockCNN
    block_cnn_batch_size = 100
    block_cnn_nb_epoch = 100
    
    """

STATIC_BASE_STRING_MNIST = """
    categorical = 0
    num_layers = 1

    num_samples = 10000
    batch_size = 100
    conditional = 0
    epochs = 300
    red_only = 1
    vae_loss_type = sad

    epsilon_std = 0.001
    training_mean = 0

    # Only for BlockCNN
    block_cnn_batch_size = 100
    block_cnn_nb_epoch = 100

    """

def calculate_total_combinations(conf_dataset):
    total_combinations = 1

    for item in conf_dataset.values():
        if type(item) == list:
            total_combinations *= len(item)

    return total_combinations


def add_str_to_file(str_to_add, file_num, dataset, block_size):

    results_dir = RESULTS_BASE_DIR + dataset+"_bs_"+str(block_size)
    intro = "# Dynamically added configurations:\n"
    open_dir = CONFIGURATION_BASE_DIR+dataset+"_bs_"+str(block_size)+".cfg"

    # Using 'a' to append
    with open(open_dir, 'a') as f:
        f.write(intro)
        f.write("results_dir = "+results_dir+"/\n")
        f.write(str_to_add)
        # f.write("block_cnn_weights = weights_block_cnn.h5\n")
        # f.write("block_vae_weights = weights_block_vae.h5\n")


def create_static_configuration(total_combinations, dataset, conf_dataset):

    to_insert_as_base = "dataset = " + dataset + "\n\n"
    for line in STATIC_BASE_STRING_LFW.splitlines():
        if line == "":
            continue
        to_insert_as_base += (line.rstrip().lstrip() + "\n")

    for i in range(total_combinations):

        block_size = conf_dataset["block_size"][0]

        open_dir = CONFIGURATION_BASE_DIR+dataset+"_bs_"+str(block_size)+".cfg"
        new_config = open(open_dir, 'w+')
        new_config.write(to_insert_as_base)


if __name__ == '__main__':

    if not CONFIGURATION_BASE_DIR.endswith("/"):
        CONFIGURATION_BASE_DIR += "/"

    if not RESULTS_BASE_DIR.endswith("/"):
        RESULTS_BASE_DIR += "/"

    if not os.path.exists(CONFIGURATION_BASE_DIR):
        os.makedirs(CONFIGURATION_BASE_DIR)

    # Add here any new variables to permute
    # new_vars = {
    #     "config_mnist_2": {
    #         "dataset": "cifar",
    #         "block_size": [],
    #         "original_dim": [196],
    #         "intermediate_dim": [147],
    #         "latent_dim": [98]
    #     }
    # }

    block_sizes = [pow(2,x) for x in range(0, 8)]

    new_vars = {}

    for i, block_size in enumerate(block_sizes):

        latent_dim = 12 * block_size
        original_dim = 24 * block_size
        intermediate_dim = 96 *block_size

        new_vars["config_lfw_"+str(i)] = {}
        config_dict = new_vars["config_lfw_"+str(i)]
        config_dict["dataset"] = "lfw"
        config_dict["block_size"] = [block_size]
        config_dict["original_dim"] = [original_dim]
        config_dict["intermediate_dim"] = [intermediate_dim]#[int(0.75 * (block_size*block_size*3))]
        config_dict["latent_dim"] = [latent_dim] #[block_size*2]


    for set, conf_dataset in enumerate(new_vars.values()):

        dataset = conf_dataset['dataset']
        del conf_dataset['dataset']

        print "Set: "+str(set)+"\nUsing dataset: " + dataset

        # Calculate the total number of combinations
        total_combinations = calculate_total_combinations(conf_dataset)

        print "Total combinations: "+str(total_combinations)
        print ""

        # Create the static configuration for each conf file
        create_static_configuration(total_combinations, dataset, conf_dataset)

        # Get all possible combinations
        all_combinations = list(itertools.product(*conf_dataset.values()))

        # Get all pairs to save them into each file
        vars_names = list(conf_dataset.keys())
        str_to_add = ""
        for i in range(len(all_combinations)):
            for j, k in zip(vars_names, all_combinations[i]):
                str_to_add += j + " = " + str(k) + "\n"
            # print str_to_add
            block_size = all_combinations[0][1]
            add_str_to_file(str_to_add, i, dataset, block_size)
            # Reset string to add
            str_to_add = ""
