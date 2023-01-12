# Pock/Chambolleの一次法

先の[鞍点問題](./separable_problem.md#鞍点問題)の数値解法として、
[参考文献](./reference.md)[1]では以下（を含めた形）のアルゴリズムが提示されている。
$$
    \begin{array}{l}
        x^{k+1}=\mathbf{prox}^{\tau}_G(x^k-\mathbf{diag}(\tau)K^Ty^k) \\
        y^{k+1}=\mathbf{prox}^{\sigma}_{F^\ast}(y^k+\mathbf{diag}(\sigma)K(2x^{k+1}-x^k))
    \end{array}
$$

$\tau\in\mathbb{R}^n_{++},\ \sigma\in\mathbb{R}^m_{++}$ であり、
$\mathbf{diag}(\tau),\ \mathbf{diag}(\sigma)$ はアルゴリズムパラメータであるとともに前処理行列としての役割をもつ。
$$
    \begin{array}{ll}
        \tau_j=\frac1{\sum_{i=1}^m|K_{i,j}|} & \quad(j=1,\ldots,n) \\
        \sigma_i=\frac1{\sum_{j=1}^n|K_{i,j}|} & \quad(i=1,\ldots,m)
    \end{array}
$$
と定めると、このとき
$$
    \|\mathbf{diag}(\sigma)^{\frac12} K \mathbf{diag}(\tau)^{\frac12}\|^2\le1
$$
（左辺のノルムは作用素ノルム）が成立する[^1]。
また、この不等式が成立するとき上記アルゴリズムが収束する（解が存在すれば）ことが示されている。

$\mathbf{prox}^\tau_G$ は近接作用素であるが、$\mathbf{diag}(\tau)^{-1}$ によりスケールされた内積に誘導されるノルム
$\|x\|_{\mathbf{diag}(\tau)^{-1}}=\langle x,x\rangle_{\mathbf{diag}(\tau)^{-1}}^{\frac12}=(x^T\mathbf{diag}(\tau)^{-1}x)^{\frac12}$
を用いて定義されている[^2]：
$$
    \mathbf{prox}^\tau_G(\tilde x) = \arg\min_x \left( G(x) + \frac12\|x-\tilde x\|_{\mathbf{diag}(\tau)^{-1}}^2 \right)
$$
なおMoreau分解により
$$
    \tilde y = \mathbf{prox}^\sigma_F(\tilde y) + \mathbf{prox}^\sigma_{F^\ast}(\tilde y)
$$
が成り立つ。


[^1]: `Totsu`では、$\mathbf{prox}$ の計算を容易にするため $\tau$ にグルーピングを施すが、この不等式が成立するようにグルーピングしている。

[^2]: `Totsu`では、対角要素グルーピングにより、スケールされたノルムを考慮しなくてよい。また $G,F$ を指示関数とするため近接作用素が単に射影となる。
