import haiku as hk
import jax.numpy as jnp
import jax
import MPNN as MPNN

class ENCODING:
    def __init__(self,  node_features,
                 edge_features, hidden_dim,
                 num_encoder_layers=3,
                  k_neighbors=48,
                 augment_eps=0.0, dropout=0.):
      super(ENCODING, self).__init__()

      self.MPNN = MPNN.ENC(node_features,edge_features,hidden_dim,num_encoder_layers,k_neighbors,augment_eps,dropout)
      self.siz = node_features
    def __call__(self,x1):
      X1,mask1,res1,ch1 = x1
      h_V1 = self.MPNN(X1,mask1,res1,ch1)
      return h_V1

class ENCODING_KMEANS:
    def __init__(self,  node_features,
                 edge_features, hidden_dim,
                 num_encoder_layers=3,
                  k_neighbors=48,nb_clusters = 20,
                 augment_eps=0.0, dropout=0.):
      super(ENCODING_KMEANS, self).__init__()

      self.MPNN = MPNN.ENC(node_features,edge_features,hidden_dim,num_encoder_layers,k_neighbors,augment_eps,dropout)
      self.siz = node_features
      self.nb_clusters = nb_clusters

    def __call__(self,x1):
      X1,mask1,res1,ch1 = x1
      h_V1 = self.MPNN(X1,mask1,res1,ch1)
      C = hk.get_parameter("centers",shape = [self.siz,self.nb_clusters],init = hk.initializers.RandomNormal(1,0))
      temp1 = jnp.einsum("nia,aj->nij",h_V1,C)
      h_V1_ = jax.nn.one_hot(temp1.argmax(-1),self.nb_clusters)
      h_V1 = jnp.einsum("nia,ja->nij",h_V1_,C)
      return h_V1,h_V1_

def encoding(x1,node_features = 64,
                 edge_features = 64, hidden_dim = 64,
                 num_encoder_layers=3,
                  k_neighbors=48,categorical = False,nb_clusters = 20):
  if categorical:
    a = ENCODING_KMEANS(node_features,edge_features,hidden_dim,num_encoder_layers,k_neighbors,nb_clusters = nb_clusters)
  else:
     a = ENCODING(node_features,edge_features,hidden_dim,num_encoder_layers,k_neighbors)
  return a(x1)
