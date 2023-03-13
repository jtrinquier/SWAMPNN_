import haiku as hk
import jax.numpy as jnp
import jax
import MPNN as MPNN

class ENCODING:
    def __init__(self,  node_features,
                 edge_features, hidden_dim,
                 num_encoder_layers=3,
                  k_neighbors=64,
                 augment_eps=0.0, dropout=0.):
      super(ENCODING, self).__init__()

      self.MPNN = MPNN.ENC(node_features,edge_features,hidden_dim,num_encoder_layers,k_neighbors,augment_eps,dropout)
      self.siz = node_features
    def __call__(self,x1):
      X1,mask1,res1,ch1 = x1
      h_V1 = self.MPNN(X1,mask1,res1,ch1)
      return h_V1

def encoding(x1,node_features = 64,
                 edge_features = 64, hidden_dim = 64,
                 num_encoder_layers=3,
                  k_neighbors=64):
  a = ENCODING(node_features,edge_features,hidden_dim,num_encoder_layers,k_neighbors)
  return a(x1)
