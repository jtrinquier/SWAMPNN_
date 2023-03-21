# SWAMPNN_
Work in progress
## Encode and Align structures
The colab notebook "ENCODE_ALIGN" gives an example of how to align structures using SWAMPNN. An example input file is provided (pdb_list_example), as well as the pdb files. 2 models are provided, one with continuous embeddings, and one with categorical embeddings with 20 clusters.
The notebook allows to get the embeddings of each position of the structures using the encoder of ProteinMPNN, and then align and score the pairs using LDDT scores.
