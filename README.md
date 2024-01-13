## Numerical Analysis for Machine Learning project | A.A. 2022-2023 | Politecnico di Milano
---
### Intoduction

This repository contains an implementation of the PageRank algorithm introduced by Larry Page and Sergey Brin. It also contains an implementation of Graph ADTs in order to make the algorithm work properly.
The aims of the project are about understanding the usage of objective-oriented programming in Python and learning how web surfers work. 
Browsers, such as Google, are used to rank nodes in the network (Graph) respect their importance. But how to do that? This is what PageRanking was developed for.
The project aims to develop a performant version in order to scale also on big datasets. 

---
### Results
In the following table are presented the performance on different datasets:

| **Dataset**          | **Node**  | **Edges** | **Time**    |
|----------------------|:---------:|:---------:|:-----------:|
| *Cartoon Characters* | 9         | 8         | 3ms         |
| *Email*              | 1k        | 26k       | 316ms       |
| *Twitter*            | 81k       | 1.8M      | 160s        |
