# ML-Class-Project

Link prediction on academic paper citation graphs.

<p align="center">
	<img src="https://upload.wikimedia.org/wikipedia/commons/c/c6/Topological_Ordering.svg" width="500">
	<em>source: Wikimedia </em>
</p>

## Usage

### Requirements

 - Python 3.8
 - Pytorch 1.11.0 & Cuda 11.3
 - Conda or Miniconda

### Setup

Run the following to set your environment up with the required packages.
```
conda create -n [ENVNAME]
source activate [ENVNAME]
conda env update --file environment.yml --prune 
```
To download and process the dataset, and generate title embeddings, run the following commands.
```
curl -f https://lfs.aminer.cn/lab-datasets/citation/DBLP-citation-network-Oct-19.tar.gz --output citation-network4.tar.gz
tar -xf citation-network4.tar.gz
python source/data_parsing.py
python source/embedding.py
```

### Running Benchmarks

Toy GNN: `python source/toy_gnn.py`
Deep GCN: `python source/deep_gcn.py`
