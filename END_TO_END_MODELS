import jax 
import jax.numpy as jnp
import haiku as hk
import MPNN
import SW


class END_TO_END:

    def __init__(self,  node_features,
                 edge_features, hidden_dim,
                 num_encoder_layers=1,
                  k_neighbors=64,
                 augment_eps=0.05, dropout=0.,affine = False):
      super(END_TO_END, self).__init__()

      self.MPNN = MPNN.ENC(node_features,edge_features,hidden_dim,num_encoder_layers,k_neighbors,augment_eps,dropout)
      self.my_sw_func = jax.jit(SW.sw(batch=True))
      self.affine  = affine
      if affine:
          self.my_sw_func = jax.jit(SW.sw_affine(batch=True))
      else:
          self.my_sw_func = jax.jit(SW.sw(batch=True))
      self.siz = node_features

    def __call__(self,x1,x2,lens,t):
      X1,mask1,res1,ch1 = x1
      X2,mask2,res2,ch2 = x2
      h_V1 = self.MPNN(X1,mask1,res1,ch1)
      h_V2 = self.MPNN(X2,mask2,res2,ch2)
      #encodings
      gap = hk.get_parameter("gap", shape=[1],init = hk.initializers.RandomNormal(0.1,-1))
      if self.affine:
          popen = hk.get_parameter("open", shape=[1],init = hk.initializers.RandomNormal(0.1,-3))
      #######
      sim_matrix = jnp.einsum("nia,nja->nij",h_V1,h_V2)
      if self.affine:
          scores,soft_aln  = self.my_sw_func(sim_matrix, lens, popen[0],gap[0],t)
      else:
          scores,soft_aln  = self.my_sw_func(sim_matrix, lens, gap[0],t)
      return soft_aln,sim_matrix,scores

class END_TO_END_SEQ_KMEANS:
    def __init__(self,  node_features,
                 edge_features, hidden_dim,
                 num_encoder_layers=1,
                  k_neighbors=64,
                 augment_eps=0.05, dropout=0.,affine = False,nb_clusters = 20):
      super(END_TO_END_SEQ_KMEANS, self).__init__()

      self.MPNN = MPNN.ENC(node_features,edge_features,hidden_dim,num_encoder_layers,k_neighbors,augment_eps,dropout)
      self.my_sw_func = jax.jit(SW.sw(batch=True))
      self.affine  = affine
      if affine:
          self.my_sw_func = jax.jit(SW.sw_affine(batch=True))
      else:
          self.my_sw_func = jax.jit(SW.sw(batch=True))
      self.siz = node_features
      self.nb_clusters = nb_clusters
    def __call__(self,x1,x2,lens,t):
      X1,mask1,res1,ch1 = x1
      X2,mask2,res2,ch2 = x2
      h_V1 = self.MPNN(X1,mask1,res1,ch1)
      h_V2 = self.MPNN(X2,mask2,res2,ch2)
      #encodings
      C = hk.get_parameter("centers",shape = [self.siz,self.nb_clusters],init = hk.initializers.RandomNormal(1,0))

      temp1 = jnp.einsum("nia,aj->nij",h_V1,C)
      temp2 = jnp.einsum("nia,aj->nij",h_V2,C)

      h_V1_ = jax.lax.stop_gradient(jax.nn.one_hot(temp1.argmax(-1),self.nb_clusters)-jax.nn.softmax(t**-1 *temp1))+jax.nn.softmax(t**-1 *temp1)
      h_V2_ = jax.lax.stop_gradient(jax.nn.one_hot(temp2.argmax(-1),self.nb_clusters)-jax.nn.softmax(t**-1 *temp2))+jax.nn.softmax(t**-1 *temp2)

      h_V1 = jnp.einsum("nia,ja->nij",h_V1_,C)
      h_V2 = jnp.einsum("nia,ja->nij",h_V2_,C)

      #kmeans
      gap = hk.get_parameter("gap", shape=[1],init = hk.initializers.RandomNormal(0.1,-1))
      if self.affine:
          popen = hk.get_parameter("open", shape=[1],init = hk.initializers.RandomNormal(0.1,-3))
      #######
      sim_matrix = jnp.einsum("nia,nja->nij",h_V1,h_V2)
      
      if self.affine:
          scores,soft_aln  = self.my_sw_func(sim_matrix, lens, popen[0],gap[0],t)
      else:
          scores,soft_aln  = self.my_sw_func(sim_matrix, lens, gap[0],t)
      return soft_aln,sim_matrix,scores,(h_V1_,h_V2_)