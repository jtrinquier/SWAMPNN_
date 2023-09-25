import jax 
import jax.numpy as jnp
import haiku as hk
import MPNN
import SW
from jax import vmap


"""
default Smith-Waterman, possibility to use softmax instead by setting soft_max = True
"""
class END_TO_END:

    def __init__(self,  node_features,
                 edge_features, hidden_dim,
                 num_encoder_layers=1,
                  k_neighbors=64,
                 augment_eps=0.05, dropout=0.,affine = False,soft_max = False,mixture = False):


      super(END_TO_END, self).__init__()

      self.MPNN = MPNN.ENC(node_features,edge_features,hidden_dim,num_encoder_layers,k_neighbors,augment_eps,dropout)
      self.my_sw_func = jax.jit(SW.sw(batch=True))
      self.affine  = affine
      if affine:
          self.my_sw_func = jax.jit(SW.sw_affine(batch=True))
      else:
          self.my_sw_func = jax.jit(SW.sw(batch=True))
      self.siz = node_features
      self.soft_max = soft_max
      self.mixture = mixture

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
      if self.soft_max == False:
        if self.affine:
            scores,soft_aln  = self.my_sw_func(sim_matrix, lens, popen[0],gap[0],t)
        else:
            scores,soft_aln  = self.my_sw_func(sim_matrix, lens, gap[0],t)
        return soft_aln,sim_matrix,scores
      
      else:
        if self.mixture == False:
            soft_aln = vmap(soft_max_single, in_axes=(0, 0, None))(sim_matrix, lens,t)
            return soft_aln,sim_matrix,0 ###TO DO: FIND A SCORE FOR THE SOFTMAX
        else:
            if self.affine:
                scores,soft_aln  = self.my_sw_func(sim_matrix, lens, popen[0],gap[0],t)
            else:
                scores,soft_aln  = self.my_sw_func(sim_matrix, lens, gap[0],t)
            soft_aln2 = vmap(soft_max_single, in_axes=(0, 0, None))(sim_matrix, lens,t)
            return soft_aln,soft_aln2,sim_matrix,scores







class END_TO_END_SEQ_KMEANS:
    def __init__(self,  node_features,
                 edge_features, hidden_dim,
                 num_encoder_layers=1,
                  k_neighbors=64,
                 augment_eps=0.05, dropout=0.,affine = False,nb_clusters = 20,soft_max = False,mixture = False):
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
      self.soft_max = soft_max
      self.mixture = mixture

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
      
      if self.soft_max == False:
        if self.affine:
            scores,soft_aln  = self.my_sw_func(sim_matrix, lens, popen[0],gap[0],t)
        else:
            scores,soft_aln  = self.my_sw_func(sim_matrix, lens, gap[0],t)
        return soft_aln,sim_matrix,scores,(h_V1_,h_V2_)
      
      else:
        if self.mixture == False:
            soft_aln = vmap(soft_max_single, in_axes=(0, 0, None))(sim_matrix, lens,t)
            return soft_aln,sim_matrix,0,(h_V1_,h_V2_) ###TO DO: FIND A SCORE FOR THE SOFTMAX
        else:
            if self.affine:
                scores,soft_aln  = self.my_sw_func(sim_matrix, lens, popen[0],gap[0],t)
            else:
                scores,soft_aln  = self.my_sw_func(sim_matrix, lens, gap[0],t)
            soft_aln2 = vmap(soft_max_single, in_axes=(0, 0, None))(sim_matrix, lens,t)
            return soft_aln,soft_aln2,sim_matrix,scores,(h_V1_,h_V2_)
    


def soft_max_single(sim_matrix, lens, t):
    """ 
    Do softmax on a single sim_matrix
    """
    max_len_1, max_len_2 = sim_matrix.shape

    mask_1 = jnp.arange(max_len_1) < lens[0]
    mask_2 = jnp.arange(max_len_2) < lens[1]

    mask = mask_1[:, None] * mask_2[None, :]
    masked_sim_matrix = jnp.where(mask, sim_matrix, -100000)

    soft_aln = jnp.sqrt(10**-9+
        jax.nn.softmax(t**-1*masked_sim_matrix, axis=-1) *
        jax.nn.softmax(t**-1*masked_sim_matrix, axis=-2)
    )
    return  soft_aln



#### TEST CONVOLUTION, work in progress

    




class END_TO_END_SEQ_KMEANS_CONVO:
    def __init__(self, node_features, edge_features, hidden_dim,
                num_encoder_layers=1, k_neighbors=64, augment_eps=0.05,
                dropout=0., affine=False, nb_clusters=20, soft_max=False,
                mixture=False, num_conv_layers=5):
        super(END_TO_END_SEQ_KMEANS_CONVO, self).__init__()

        self.MPNN = MPNN.ENC(node_features, edge_features, hidden_dim,
                            num_encoder_layers, k_neighbors, augment_eps, dropout)
        self.my_sw_func = jax.jit(SW.sw(batch=True))
        self.affine = affine
        if affine:
            self.my_sw_func = jax.jit(SW.sw_affine(batch=True))
        else:
            self.my_sw_func = jax.jit(SW.sw(batch=True))
        self.siz = node_features
        self.nb_clusters = nb_clusters
        self.conv_layers = []

        for i in range(num_conv_layers):
            conv_layer = hk.Conv2D(output_channels=(i+1)*2, kernel_shape=(3, 3), padding="SAME")
            self.conv_layers.append(conv_layer)
        self.conv_last_layer = hk.Conv2D(output_channels=1, kernel_shape=(3, 3), padding="SAME")
        self.soft_max = soft_max
        self.mixture = mixture

def __call__(self, x1, x2, lens, t):
    X1, mask1, res1, ch1 = x1
    X2, mask2, res2, ch2 = x2
    h_V1 = self.MPNN(X1, mask1, res1, ch1)
    h_V2 = self.MPNN(X2, mask2, res2, ch2)
    gap = hk.get_parameter("gap", shape=[1], init=hk.initializers.RandomNormal(0.1, -1))
    if self.affine:
        popen = hk.get_parameter("open", shape=[1], init=hk.initializers.RandomNormal(0.1, -3))
    C = hk.get_parameter("centers", shape=[self.siz, self.nb_clusters], init=hk.initializers.RandomNormal(1, 0))

    temp1 = jnp.einsum("nia,aj->nij", h_V1, C)
    temp2 = jnp.einsum("nia,aj->nij", h_V2, C)

    h_V1_ = jax.lax.stop_gradient(jax.nn.one_hot(temp1.argmax(-1), self.nb_clusters) - jax.nn.softmax(t**-1 * temp1)) + jax.nn.softmax(t**-1 * temp1)
    h_V2_ = jax.lax.stop_gradient(jax.nn.one_hot(temp2.argmax(-1), self.nb_clusters) - jax.nn.softmax(t**-1 * temp2)) + jax.nn.softmax(t**-1 * temp2)

    h_V1 = jnp.einsum("nia,ja->nij", h_V1_, C)
    h_V2 = jnp.einsum("nia,ja->nij", h_V2_, C)

    sim_matrix = jnp.einsum("nia,nja->nij", h_V1, h_V2)


    if self.soft_max == False:
        if self.affine:
            scores,soft_aln  = self.my_sw_func(sim_matrix, lens, popen[0],gap[0],t)
        else:
            scores,soft_aln  = self.my_sw_func(sim_matrix, lens, gap[0],t)
        return soft_aln,sim_matrix,scores,(h_V1_,h_V2_)
    
    else:
        sim_matrix_conv =   sim_matrix[:,:,:,None]
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            sim_matrix_conv = conv_layer(sim_matrix_conv)
            sim_matrix_conv = jax.nn.relu(sim_matrix_conv)
        sim_matrix_conv = self.conv_last_layer(sim_matrix_conv)
        if self.mixture == False:
            soft_aln = vmap(soft_max_single, in_axes=(0, 0, None))(sim_matrix_conv[:,:,:,0], lens,t)
            return soft_aln,sim_matrix,0,(h_V1_,h_V2_) ###TO DO: FIND A SCORE FOR THE SOFTMAX
        else:
            if self.affine:
                scores,soft_aln  = self.my_sw_func(sim_matrix, lens, popen[0],gap[0],t)
            else:
                scores,soft_aln  = self.my_sw_func(sim_matrix, lens, gap[0],t)
            soft_aln2 = vmap(soft_max_single, in_axes=(0, 0, None))(sim_matrix_conv[:,:,:,0], lens,t)
            return soft_aln,soft_aln2,sim_matrix,scores,(h_V1_,h_V2_)
