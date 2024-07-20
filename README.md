## Numerical Analysis for Machine Learning project | A.A. 2022-2023 | Politecnico di Milano
---
### Intoduction

This repository contains an implementation of the PageRank algorithm introduced by Larry Page and Sergey Brin. It also contains an implementation of Graph ADTs to make the algorithm work properly.
The project aims to understand the usage of objective-oriented programming in Python and learn how web surfers work. 
Browsers, such as Google, are used to rank nodes in the network (Graph) and respect their importance. But how to do that? This is what PageRank was developed for.
The project aims to develop a performant version to scale also on big datasets. 

---
### Repository Structure
```plaintext
│   README.md
│   LICENSE
│
├───code                                            
│   ├───Makefile                                    // Relational Causal Discovery algorithm
│   ├───graph.py                                    // Class Graph Object
│   ├───graph_test.py                               
│   ├───pagerank.py                                 // PageRank implementation
│   ├───requirements.txt                            // Modules required to run the code
│   └───dataset                                     
│       ├───characters                              // Small Dataset
│       ├───email                                   // Medium Dataset
│       └───twitter                                 // Big Dataset
│   
├───deliverables                                
│   ├───NAML.pdf                                   // Slides for the oral presentation
│   └───NAML_Report.pdf                            // Project Report
│
└───plot
    └───plot.ipynb.py                              // Notebook to get the plots of the report

```

---
### Results
The following table presents the performance on different datasets:

| **Dataset**          | **Node**  | **Edges** | **Time**    |
|----------------------|:---------:|:---------:|:-----------:|
| *Cartoon Characters* | 9         | 8         | 3ms         |
| *Email*              | 1k        | 26k       | 316ms       |
| *Twitter*            | 81k       | 1.8M      | 160s        |
