import torch


def _make_mlp(input_dim, hidden_dims, output_dim, dtype=torch.float64, layer_norm=0):

    mlp_in_dims = [input_dim]
    if(hidden_dims!=""):
        mlp_in_dims = mlp_in_dims + [int(i) for i in hidden_dims.split("-")]

    mlp_out_dims = [output_dim]
    if(hidden_dims != ""):
        mlp_out_dims =  [int(i) for i in hidden_dims.split("-")] + mlp_out_dims
   
    nn_list = []

    for i in range(len(mlp_in_dims)):
       
        l = torch.nn.Linear(mlp_in_dims[i], mlp_out_dims[i], dtype=dtype)
       
        nn_list.append(l)
        
        if i < (len(mlp_in_dims) - 1):
            if(layer_norm):
                nn_list.append(torch.nn.LayerNorm(mlp_out_dims[i]))
            nn_list.append(torch.nn.Tanh())
    
    return torch.nn.Sequential(*nn_list)

class SkipMLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, dtype=torch.float64, add_skip_connection=False, layer_norm=0):
        super(SkipMLP, self).__init__()

        self.mlp=_make_mlp(input_dim,hidden_dims,output_dim, dtype=dtype, layer_norm=layer_norm)

        self.skip_connection=None
        if(add_skip_connection):
            
            self.skip_connection=torch.nn.Linear(input_dim, output_dim, dtype=dtype, bias=False)

    def forward(self, x):

        if(self.skip_connection is not None):
           
            return self.skip_connection(x)+self.mlp(x)

        else:
        
            return self.mlp(x)