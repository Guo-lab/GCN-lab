# GCN-lab  
- ## [Learn the Open Course](http://web.stanford.edu/class/cs224w/)  

- ## [Video on Bilibili](https://www.bilibili.com/video/BV1Xr4y1q788?spm_id_from=333.337.search-card.all.click&vd_source=b4c6848f0ca53aaf723f77170427ce41)  




- ## [Coding Reference](https://gitcode.net/mirrors/PolarisRisingWar/cs224w-2021-winter-colab?utm_source=csdn_github_accelerator)

---



# Checklist:  
| Slides  |  Colab |
| ----------- | ----------- |
| [lecture 1](http://web.stanford.edu/class/cs224w/slides/01-intro.pdf) | /  |
| [lecture 2](http://web.stanford.edu/class/cs224w/slides/02-tradition-ml.pdf) |  Colab-1   |
| [lecture 3](http://web.stanford.edu/class/cs224w/slides/03-nodeemb.pdf) | / |


<br><br>

---

# About Features (in lecture 2)

- ## Node-level feature: (Structures and Position)
  - ### Node degree (importance features and structure-based feature)
  - ### Node Centrality (importance features)
    - #### Engienvector Centrality
    - #### Betweenness Centrality
    - #### Closeness Centrality ...
  - ### Clustering Coefficient (structure-based feature)
  - ### Graphlets GDV (structure-based feature) 


- ## Link-level feature:
  node level features miss information of the relationship of nodes
  - ### Distance-based feature
  - ### Local neighborhood overlap
    - #### Common neighbors
    - #### Jaccard's coefficient (normalize by degree) 
    - #### Adamic-Adar index (ordinary neighbors)
  - ### Global neighborhood overlap
    - #### Katz index (number of paths)
  
- ## Graph-level Kernels: (similarity between graphs) 
    > <font style="color: rgb(250,250,0)">Design graph feature vector</font>
  - #### Graphlet Kernel (bag-of-graphlets representation)
  - #### Weisfeiler-Lehman Kernel (bag-of-node-degrees representation)
    Color Refinement & Hash aggregated colors
  - #### Random-walk Kernel
  - #### Shortest-path graph Kernel ...

<br><br>

---

# About GNN Outputs

<img src="md-img/image-20221008002758498.png" alt="md-img/image-20221008002758498" style="zoom:50%;" />


<br><br><br>

---

# Embedding, Representation Learning
> Learn the embedding is unsupervised
- ## Random Walk
  
  <img src="md-img/image-20221009005529389.png" alt="image-20221009005529389" style="zoom:28%;" align='left'/>
  <br><br><br><br>  
  Most likely be the neighbor (should be because of their cos similarity) 
  <br>
  <br>
  
  <img src="md-img/image-20221009005422630.png" alt="image-20221009005422630" style="zoom:25%;" align = 'left'/>
  
  <br><br>
  <br>

  - ### DeepWalk
  - ### Node2Vec
  - ### biased random walk based on attributes, learnd weights
  - ### To embed the nodes.
    - Embed nodes and **sum/average** them
    - Super-node spans (sub)graph, then embed that node
    - Anonymous Walk Embeddings(times each anonymous walk happens or concatenate anonymous walks embeddings)