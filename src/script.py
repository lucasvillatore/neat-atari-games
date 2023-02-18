import os
import shutil

def create_neat_config(src, parameters):
    neat_config = f"""
    [NEAT]
    fitness_criterion     = mean
    fitness_threshold     = 10000
    pop_size              = {parameters["pop_size"]}
    reset_on_extinction   = False

    [DefaultGenome]
    # node activation options
    activation_default      = {parameters["activation_default"]}
    activation_mutate_rate  = 0.0
    activation_options      = relu

    # node aggregation options
    aggregation_default     = sum
    aggregation_mutate_rate = 0.0
    aggregation_options     = sum

    # node bias options
    bias_init_mean          = 0.0
    bias_init_stdev         = 1.0
    bias_max_value          = 30.0
    bias_min_value          = -30.0
    bias_mutate_power       = 0.5
    bias_mutate_rate        = 0.7
    bias_replace_rate       = 0.1

    # genome compatibility options
    compatibility_disjoint_coefficient = 1.0
    compatibility_weight_coefficient   = 0.5

    # connection add/remove rates
    conn_add_prob           = 0.5
    conn_delete_prob        = 0.5

    # connection enable options
    enabled_default         = True
    enabled_mutate_rate     = 0.35

    feed_forward            = True
    initial_connection      = {parameters["initial_connection"]}

    # node add/remove rates
    node_add_prob           = 0.4
    node_delete_prob        = 0.4

    # network parameters
    num_hidden              = {parameters["num_hidden"]}
    num_inputs              = 3
    num_outputs             = 4

    # node response options
    response_init_mean      = 1.0
    response_init_stdev     = 0.0
    response_max_value      = 30.0
    response_min_value      = -30.0
    response_mutate_power   = 0.0
    response_mutate_rate    = 0.0
    response_replace_rate   = 0.0

    # connection weight options
    weight_init_mean        = 0.0
    weight_init_stdev       = 1.0
    weight_max_value        = 30
    weight_min_value        = -30
    weight_mutate_power     = 0.5
    weight_mutate_rate      = 0.8
    weight_replace_rate     = 0.1

    [DefaultSpeciesSet]
    compatibility_threshold = {parameters["compatibility_threshold"]}

    [DefaultStagnation]
    species_fitness_func = mean
    max_stagnation       = 20
    species_elitism      = 1

    [DefaultReproduction]
    elitism            = 3
    survival_threshold = 0.2
    """

    with open(f"{src}/neat-config", "w") as file:
        file.write(neat_config)

base = {
    "pop_size": 150,
    "initial_connection": "unconnected",
    "num_hidden": 0,
    "compatibility_threshold": 2.0,
    "activation_default": "relu"
}

train = {
    "pop_size": [
        50, 150, 300, 600
    ],
    "initial_connection": [
        "full_direct", "unconnected"
    ],
    "num_hidden": [0, 3, 6, 9],
    "compatibility_threshold":  [1.0, 2.0, 3.0, 4.0, 6.0],
    "activation_default": ["sigmoid", "relu", "tanh", "cube", "exp"]
}

def create_folder(key, value):
    
    dirs = ["checkpoints", "graphs", "network", "statistics"]

    destination = f"./pong/pong_{key}_{value}"
    os.mkdir(destination)
    for dir_tmp in dirs:
        source = f"./pong/pong_base/{dir_tmp}/.gitkeep"
        os.mkdir(destination + "/" + dir_tmp)

        shutil.copy(source, destination + "/" + dir_tmp + "/.gitkeep")
    return destination

if __name__ == '__main__':
    
    
    for key, values in train.items():

        for value in values:
            valores_teste = base.copy()

            valores_teste[key] = value
            src = create_folder(key, value)
            create_neat_config(src, valores_teste)
            # exit()
            # shutil.copy("./pong/pong_base", "./pong/pong_{}_{}".format(key, value))
            # print("Testando {}".format(valores_teste))