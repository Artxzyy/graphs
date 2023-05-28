# graphs

Implementation of the practical project for the subject of Graph Theory and Computability - Computer Science | PUC Minas

## Library

As the first part of this project, a Graph library for Python programming language was implemented. The library contains:

* Adjacency List and Adjacency Matrix with CRUD methods
    - Add Vertices and Edges
    - Remove Vertices and Edges
    - Get Vertices and Edges
    - Update Vertices and Edges
    - Write to CSV file, using [GEPHI CSV format](https://gephi.org/users/supported-graph-formats/csv-format/)
    - Write to GDF file, using [GEPHI GDF format](https://gephi.org/users/supported-graph-formats/gdf-format/)
* General methods to get Graph attributes and caracteristics
    - Print Graph
    - Test if Vertices are adjacent
    - Test if Graph is complete or empty
    - Test if Graph has Vertex or Edge labels
    - Test if Graph has Vertex of Edge weights
    - Get amount of Vertices and Edges

At the moment, the library implementations are somewhat inconsistent internally, with some methods being implemented with very different patterns. 

## Transductive Inference

As the second part of this project, it was necessary to create a semi-supervised machine learning method, explained in the article "Learning with Local and Global Consistency" and apply it to the Graphs library to be able to use the Transductive Inference in problems that can be modeled using Graph Theory.

## Reference

Zhou, Dengyong, et al. "Learning with local and global consistency." Advances in neural information processing systems 16 (2003).
