import os
from typing import Optional
from torch import load
from torch import zeros, concat 
from torch import Tensor  
from torch.nn import Module, Parameter, ModuleList
from torch.nn import Linear, LayerNorm, Conv2d
from torch.nn.functional import gelu
from torch.nn.functional import scaled_dot_product_attention as attention
from torch.hub import load_state_dict_from_url 
from model.settings import Settings
  
class Attention(Module): 
    def __init__(
        self, 
        model_dimension: int, 
        number_of_heads: int
    )-> None:
        super().__init__()
        self.norm = LayerNorm(model_dimension, eps=1e-6)
        self.q_projector = Linear(model_dimension, model_dimension)
        self.k_projector = Linear(model_dimension, model_dimension)
        self.v_projector = Linear(model_dimension, model_dimension)  
        self.o_projector = Linear(model_dimension, model_dimension)
        
        self.number_of_heads = number_of_heads
        self.scores = None 

    def split(self, sequence: Tensor) -> Tensor:
        batch_size, sequence_lenght, model_dimension = sequence.shape 
        return sequence.view(batch_size, sequence_lenght, self.number_of_heads, model_dimension // self.number_of_heads).transpose(1, 2)
 
    def merge(self, sequence: Tensor) -> Tensor: 
        sequence = sequence.transpose(1, 2)
        return sequence.reshape(*sequence.shape[:-2], -1) 

    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        sequence = self.norm(sequence)
        q, k, v = self.q_projector(sequence), self.k_projector(sequence), self.v_projector(sequence)
        q, k, v = self.split(q), self.split(k), self.split(v)
 
        if mask is not None:
            mask = mask[:, None, None, :].bool() 

        sequence = attention(q, k, v, ~mask if mask is not None else None)
        return self.o_projector(self.merge(sequence))  


class FFN(Module): 
    def __init__(
        self,
        model_dimension : int, 
        hidden_dimension: int
    )-> None:
        super().__init__()
        self.norm = LayerNorm(model_dimension, eps=1e-6)
        self.input_layer = Linear(model_dimension, hidden_dimension)
        self.output_layer = Linear(hidden_dimension, model_dimension)

    def forward(self, features: Tensor) -> Tensor: 
        features = self.norm(features)
        features = gelu(self.input_layer(features))
        return self.output_layer(features)


class Encoder(Module): 
    def __init__(
        self, 
        model_dimension : int, 
        number_of_heads : int, 
        hidden_dimension: int, 
    )-> None:
        super().__init__()
        self.attention = Attention(model_dimension, number_of_heads)
        self.ffn = FFN(model_dimension, hidden_dimension) 

    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor: 
        sequence = sequence + self.attention(sequence, mask) 
        sequence = sequence + self.ffn(sequence) 
        return sequence


class Transformer(Module): 
    def __init__(
        self, 
        number_of_layers: int, 
        model_dimension : int, 
        number_of_heads : int, 
        hidden_dimension: int
    )-> None:
        super().__init__()
        self.encoders = ModuleList([
            Encoder(model_dimension, number_of_heads, hidden_dimension) for _ in range(number_of_layers)
        ])

    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for encoder in self.encoders:
            sequence = encoder(sequence, mask)
        return sequence
    

class Positional(Module): 
    def __init__(
        self, 
        sequence_lenght: int,
        model_dimension: int
    )-> None:
        super().__init__()
        self.embeddings = Parameter(zeros(1, sequence_lenght, model_dimension))
    
    def forward(self, sequence: Tensor) -> Tensor: 
        return sequence + self.embeddings
    

class Labeler(Module):
    def __init__(
        self, 
        model_dimension: int
    )-> None:
        super().__init__()
        self.embeddings = Parameter(zeros(1, 1, model_dimension)) 

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = sequence.flatten(2).transpose(1, 2) 
        return concat((self.embeddings.expand(sequence.size(0), -1, -1), sequence), dim=1) 
    

class ViT(Module):  
    def __init__(
        self,  
        image_size: tuple[int, int], 
        patch_size: tuple[int, int],
        number_of_classes: int = 1000,
        model_dimension  : int = 768,
        hidden_dimension : int = 3072,
        number_of_heads  : int = 12,
        number_of_layers : int = 12  
    )-> None:
        super().__init__()    
        self.image_size = image_size                 
        h, w = image_size
        fh, fw = patch_size
        gh, gw = h // fh, w // fw    
 
        self.patcher = Conv2d(3, model_dimension, kernel_size=(fh, fw), stride=(fh, fw))
        self.labeler = Labeler(model_dimension) 
        self.positional = Positional(gh * gw + 1, model_dimension)
         
        self.transformer = Transformer(
            number_of_layers=number_of_layers, 
            model_dimension=model_dimension, 
            number_of_heads=number_of_heads, 
            hidden_dimension=hidden_dimension
        ) 
 
        self.norm = LayerNorm(model_dimension, eps=1e-6)
        self.head = Linear(model_dimension, number_of_classes)  
          
    def forward(self, sequence: Tensor) -> Tensor:  
        if sequence.dim() == 3:
            sequence = sequence.unsqueeze(0)
        sequence = self.patcher(sequence)  
        sequence = self.labeler(sequence)  
        sequence = self.positional(sequence)   
        sequence = self.transformer(sequence) 
        sequence = self.norm(sequence)[:, 0]  
        sequence = self.head(sequence)  
        return sequence 
    
    def initialize(self, name: str, path: Optional[str] = None, location: str = 'cpu'):
        if path is not None and os.path.exists(path):
            state_dict = load(path, map_location=location)
        else:  
            base = "https://huggingface.co/eric-hermosis/ViT-Imagenet1k/resolve/main/"
            url = os.path.join(base, f"{name}.pth") 
            state_dict = load_state_dict_from_url(url, map_location=location)
        self.load_state_dict(state_dict) 

    @staticmethod
    def build(name: str) -> ViT:
        settings = Settings.get(name)
        vit = ViT(**settings.dump())
        vit.initialize(name)
        return vit