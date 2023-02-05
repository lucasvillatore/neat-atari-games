#!/bin/python3
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("target", help="target problem")
parser.add_argument("source", help="source problem")
args = parser.parse_args()

def create_transfer_learning(name, source):
    
    create_folder(name)
    create_checkpoints(name)
    create_graphs(name)
    create_network(name)
    create_statistics(name)
    create_neat_config(name, source)

def create_checkpoints(target):
    create_folder(f"{target}/checkpoints")
    create_file(f"{target}/checkpoints/.gitkeep")

def create_graphs(target):
    create_folder(f"{target}/graphs")
    create_file(f"{target}/graphs/.gitkeep")

def create_network(target):
    create_folder(f"{target}/network")
    create_file(f"{target}/network/.gitkeep")

def create_statistics(target):
    create_folder(f"{target}/statistics")
    create_file(f"{target}/statistics/.gitkeep")

def create_neat_config(target, source):
    file_content = None
    try:
        with open(f"{source}/neat-config", "r") as config:
            file_content = config.read()
    except Exception as err:
        print("error when creating neat-config - Creating configuration file empty")

    create_file(f"{target}/neat-config", file_content)

def create_folder(name):
    print(f"Creating {name}")
    os.mkdir(name)

def create_file(name, content = None):
    print(f"Creating {name}")
    with open(name, "w") as file:
        if content is not None:
            file.write(content)

def validate_args(args):

    target = args.target
    source = args.source

    if not os.path.exists(target):
        print(f"target {target} not exists")
        exit()

    if not os.path.exists(source):
        print(f"source {source} not exists")
        exit()
    
def new_folder_transfer_learning(target, source):
    source = source.replace("./", "")
    source = source.split("/")[1]

    return f"{target}/{target}_{source}"
    
if __name__ == '__main__':
    validate_args(args)

    target = args.target
    source = args.source

    new_transfer_learning = new_folder_transfer_learning(target, source)

    create_transfer_learning(new_transfer_learning, source)
