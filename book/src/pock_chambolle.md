# Pock/Chambolleの一次法

先の[鞍点問題](./separable_problem.md#鞍点問題)の数値解法として、
[参考文献](./reference.md)[1]では以下（を含めた形）のアルゴリズムが提示されている。
\\[
    \begin{array}{l}
        x^{k+1}={\bf prox}^{\tau}\_G(x^k-{\bf diag}(\tau)K^Ty^k) \\\\
        y^{k+1}={\bf prox}^{\sigma}\_{F^\ast}(y^k+{\bf diag}(\sigma)K(2x^{k+1}-x^k))
    \end{array}
\\]

\\(\tau\in{\bf R}^n\_{++},\ \sigma\in{\bf R}^m\_{++}\\) であり、
\\({\bf diag}(\tau),\ {\bf diag}(\sigma)\\) はアルゴリズムパラメータであるとともに前処理行列としての役割をもつ。
\\[
    \begin{array}{l}
        \tau_j=\frac1{\sum_{i=1}^m|K_{i,j}|} & \quad(j=1,\ldots,n) \\\\
        \sigma_i=\frac1{\sum_{j=1}^n|K_{i,j}|} & \quad(i=1,\ldots,m)
    \end{array}
\\]
と定めると、このとき
\\[
    \\|{\bf diag}(\sigma)^{\frac12} K {\bf diag}(\tau)^{\frac12}\\|^2\le1
\\]
（左辺のノルムは作用素ノルム）が成立する[^totsu2]。
また、この不等式が成立するとき上記アルゴリズムが収束する（解が存在すれば）ことが示されている。

\\({\bf prox}^\tau\_G\\) は近接作用素であるが、\\({\bf diag}(\tau)^{-1}\\) によりスケールされた内積に誘導されるノルム
\\(\\|x\\|\_{{\bf diag}(\tau)^{-1}}=\langle x,x\rangle\_{{\bf diag}(\tau)^{-1}}^{\frac12}=(x^T{\bf diag}(\tau)^{-1}x)^{\frac12}\\)
を用いて定義されている[^totsu3]：
\\[
    {\bf prox}^\tau\_G(\tilde x) = \arg\min_x \left( G(x) + \frac12\\|x-\tilde x\\|\_{{\bf diag}(\tau)^{-1}}^2 \right)
\\]
なおMoreau分解により
\\[
    \tilde y = {\bf prox}^\sigma\_F(\tilde y) + {\bf prox}^\sigma\_{F^\ast}(\tilde y)
\\]
が成り立つ。

---

[^totsu2]: `Totsu`では、\\({\bf prox}\\) の計算を容易にするため \\(\tau\\) にグルーピングを施すが、この不等式が成立するようにグルーピングしている。

[^totsu3]: `Totsu`では、対角要素グルーピングにより、スケールされたノルムを考慮しなくてよい。また \\(G,F\\) を指示関数とするため近接作用素が単に射影となる。
