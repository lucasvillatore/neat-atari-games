class InterfaceGames:

    def __init__(self, name, neat_config_path, folder, checkpoint = None) -> None:
        self.name = name
        self.neat_config_path = neat_config_path
        self.folder = folder
        self.checkpoint = checkpoint