# GCN-lab  
- ## [Learn the Open Course](http://web.stanford.edu/class/cs224w/)  

- ## [Video on Bilibili](https://www.bilibili.com/video/BV1Xr4y1q788?spm_id_from=333.337.search-card.all.click&vd_source=b4c6848f0ca53aaf723f77170427ce41)  




- ## [Coding Reference](https://gitcode.net/mirrors/PolarisRisingWar/cs224w-2021-winter-colab?utm_source=csdn_github_accelerator)

---



# Checklist:  
| Slides  |  Colab |
| ----------- | ----------- |
| [lecture 1](http://web.stanford.edu/class/cs224w/slides/01-intro.pdf) | /  |
| [lecture 2](http://web.stanford.edu/class/cs224w/slides/02-tradition-ml.pdf) |  Colab   |


# About features

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
  - ### Graphlet Kernel
  - ### Weisfeiler-Lehman Kernel
  - ### Random-walk Kernel
  - ### Shortest-path graph Kernel ...