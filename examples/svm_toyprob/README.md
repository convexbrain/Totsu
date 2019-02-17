# Example: svm_toyprob

* Hard-margin SVM with a gaussian kernel
* Trained by 2-dimensional ring-shaped labeled points
* Classifying whether a point is inside of the ring or not

## Running this Example

```
$ cargo run --release
$ python plot.py
```

![](plot.png)

* Larger points: support vectors
* Curved lines: clasification boundary

## Formulation

* <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{array}{ll}&space;\text{&space;minimize&space;}_\alpha&space;&&space;\frac12\sum_{i,j}&space;\alpha_i\alpha_jy_iy_jK(x_i,x_j)&space;-&space;\sum_i\alpha_i&space;\\&space;\text{&space;subject&space;to&space;}&space;&&space;\sum_i&space;\alpha_i&space;y_i&space;=&space;0&space;\\&space;&&space;\alpha_i&space;\ge&space;0&space;\quad&space;(i=1,\ldots,l)&space;\end{array}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{array}{ll}&space;\text{&space;minimize&space;}_\alpha&space;&&space;\frac12\sum_{i,j}&space;\alpha_i\alpha_jy_iy_jK(x_i,x_j)&space;-&space;\sum_i\alpha_i&space;\\&space;\text{&space;subject&space;to&space;}&space;&&space;\sum_i&space;\alpha_i&space;y_i&space;=&space;0&space;\\&space;&&space;\alpha_i&space;\ge&space;0&space;\quad&space;(i=1,\ldots,l)&space;\end{array}" title="\begin{array}{ll} \text{ minimize }_\alpha & \frac12\sum_{i,j} \alpha_i\alpha_jy_iy_jK(x_i,x_j) - \sum_i\alpha_i \\ \text{ subject to } & \sum_i \alpha_i y_i = 0 \\ & \alpha_i \ge 0 \quad (i=1,\ldots,l) \end{array}" align="top" /></a>
* Solved by QP
