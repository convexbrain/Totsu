# Example: imgnr_udef

* Image noise reduction by sparse Laplacian
* Constraining L1-norm of Laplacian image
* Minimizing Euclidean distance from noisy input image

## Running this Example

```
$ cargo run --release
```

![](miku_face_noise.png)
![](out.png)

* Left: input image, right: output image
* Hatsune Miku by Crypton Future Media, INC. 2007 is licensed under a Creative Commons Attribution-NonCommercial 3.0 Unported License. Based on a work at https://piapro.net/license.

## Formulation

* <img src="https://latex.codecogs.com/svg.image?\begin{array}{ll}\text{minimize}_{x,z,t}&space;&&space;z&space;\\\text{subject&space;to}&&space;-t&space;\le&space;Lx&space;\le&space;t&space;\\&&space;\mathbf{1}^Tt&space;\le&space;\lambda_\text{ratio}&space;\\&&space;0&space;\le&space;x&space;\le&space;1&space;\\&&space;\|x-\hat&space;x\|&space;\le&space;z\end{array}" title="\begin{array}{ll}\text{minimize}_{x,z,t} & z \\\text{subject to}& -t \le Lx \le t \\& \mathbf{1}^Tt \le \lambda_\text{ratio} \\& 0 \le x \le 1 \\& \|x-\hat x\| \le z\end{array}" align="top" />
* Solved by user-defined problem implementation which efficiently combines LP and SOCP.
