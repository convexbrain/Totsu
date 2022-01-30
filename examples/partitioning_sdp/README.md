# Example: partitioning_sdp

* Divide nodes of a 2-dimensional planar graph into two groups
* Graph edge weights represent the signed distance  (positive: dissimilarity, negative: similarity) between nodes
* Find a maximum cut (maximum sum of edge weights between nodes of different classification) approximately

## Running this Example

```
$ cargo run --release
```

![](plot.svg)

* Red lines: positive edge weights, the thicker the more dissimilar
* Blue lines: negative edge weights, the thicker the more similar
* Black and white circles: represent the classification of nodes

## Formulation

* <img src="https://latex.codecogs.com/svg.image?\begin{array}{ll}&space;\text{&space;minimize&space;}_{X}&space;&&space;\mathbf{Tr}(WX)&space;\\&space;\text{&space;subject&space;to&space;}&space;&&space;X&space;\succeq&space;0&space;\\&space;&&space;X_{ii}=1&space;\quad&space;(i=1,\ldots,l)&space;\end{array}" title="\begin{array}{ll} \text{ minimize }_{X} & \mathbf{Tr}(WX) \\ \text{ subject to } & X \succeq 0 \\ & X_{ii}=1 \quad (i=1,\ldots,l) \end{array}" align="top" />
* Solved by SDP
