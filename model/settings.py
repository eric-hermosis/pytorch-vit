from yaml import safe_load
from dataclasses import dataclass, asdict

@dataclass
class Settings:
    patch_size: list[int]
    image_size: list[int]
    model_dimension  : int
    hidden_dimension : int
    number_of_layers : int
    number_of_heads  : int
    number_of_classes: int

    @classmethod
    def get(cls, name: str, path: str = 'settings.yaml') -> Settings:
        configurations: dict
        with open(path, "r") as file:
            configurations = safe_load(file)   
        
        configuration = configurations[name]
        return cls(
            patch_size=configuration["patch_size"],
            image_size=configuration["image_size"],
            model_dimension=configuration["model_dimension"],
            hidden_dimension=configuration["hidden_dimension"],
            number_of_layers=configuration["number_of_layers"],
            number_of_heads=configuration["number_of_heads"],
            number_of_classes=configuration["number_of_classes"],
        ) 
    
    def dump(self) -> dict:
        return asdict(self)

