# 二手法の組合せ

`Totsu`では、前述した二手法を組み合わせて、錐線形計画問題を解く。

まず[Homogeneous Self-dual Embedding](./selfdual_embed.md)を施した最適性条件を、少し変数名と形を変えて、以下に再掲する：
\\[
    \left[ \begin{matrix}
    0 & A^T & 0 & c\\\\
    -A & 0 & -I & b \\\\
    -c^T & -b^T & 0 & 0
    \end{matrix} \right]
    \left[ \begin{matrix}
    \hat x \\\\ \hat y \\\\ \hat s \\\\ \hat \tau
    \end{matrix} \right]
    \in
    \left[ \begin{matrix}
    \lbrace0\rbrace^n \\\\ \lbrace0\rbrace^m \\\\ {\bf R}\_+
    \end{matrix} \right]
    , \qquad
    \hat x \in {\bf R}^n, \quad
    \hat y \in \mathcal{K}^*, \quad
    \hat s \in \mathcal{K}, \quad
    \hat \tau \in {\bf R}\_+
\\]
そして
\\[
    K=\left[ \begin{matrix}
    0 & A^T & 0 & c\\\\
    -A & 0 & -I & b \\\\
    -c^T & -b^T & 0 & 0
    \end{matrix} \right], \qquad
    x=\left[ \begin{matrix}
    \hat x \\\\ \hat y \\\\ \hat s \\\\ \hat \tau
    \end{matrix} \right]
\\]
とおく。

\\(G\\) を \\(\hat x \in {\bf R}^n,\ \hat y \in \mathcal{K}^\ast,\ \hat s \in \mathcal{K},\ \hat \tau \in {\bf R}\_+\\) の指示関数とする：
\\[
    G(x)=
    I_{{\bf R}^n \times \mathcal{K}^\ast \times \mathcal{K} \times {\bf R}\_+}(x)=
    \left\lbrace \begin{array}{l}
    0 & ({\rm if}\ x \in {\bf R}^n \times \mathcal{K}^\ast \times \mathcal{K} \times {\bf R}\_+) \\\\
    +\infty & ({\rm otherwise})
    \end{array} \right.
\\]
同様に \\(F\\) も指示関数
\\[
    F(y)=
    I_{\lbrace0\rbrace^{n+m} \times {\bf R}\_+}(y)=
    \left\lbrace \begin{array}{l}
    0 & ({\rm if}\ y \in \lbrace0\rbrace^{n+m} \times {\bf R}\_+) \\\\
    +\infty & ({\rm otherwise})
    \end{array} \right.
\\]
とすることで、元の式を
\\[
    \begin{array}{l}
    {\rm minimize} & G(x) + F(Kx)
    \end{array}
\\]
と表すことができる。
これに[Pock/Chambolleの一次法](./pock_chambolle.md)を適用すればよい。

## 近接作用素と前処理行列

まず
\\[
    \begin{array}{l}
    {\bf prox}^\sigma\_{F^\ast}(\tilde y)
    &=& \tilde y - {\bf prox}^\sigma\_F(\tilde y) \\\\
    &=& \tilde y - \arg\min_y \left( F(y) + \frac12\\|y-\tilde y\\|\_{{\bf diag}(\sigma)^{-1}}^2 \right)
    \end{array}
\\]
ここで \\(F\\) は指示関数なので \\(\arg\min\\) はその集合への射影となるが、
一般には \\({\bf diag}(\sigma)^{-1}\\) によってスケールされた距離にもとづく必要がある。
しかし \\(\lbrace0\rbrace^{n+m} \times {\bf R}\_+\\) への射影は各要素で互いに独立に分解してよく、
結局 \\(\sigma\\) に依存せずに
\\[
    \begin{array}{l}
    {\bf prox}^\sigma\_{F^\ast}(\tilde y)
    &=& \tilde y - \Pi_{\lbrace0\rbrace^{n+m} \times {\bf R}\_+} (\tilde y) \\\\
    &=& \tilde y - \left[ \begin{matrix}
                   \lbrace0\rbrace^{n+m} \\\\ \max(\tilde y\_{n+m+1}, 0)
                   \end{matrix} \right] \\\\
    &=& \left[ \begin{matrix}
        \tilde y_1 \\\\ \vdots \\\\ \tilde y_{n+m} \\\\ \min(\tilde y_{n+m+1}, 0)
        \end{matrix} \right] \\\\
    \end{array}
\\]
となる。

次に
\\[
    \begin{array}{l}
    {\bf prox}^\tau\_{G}(\tilde x)
    &=& \arg\min_x \left( G(x) + \frac12\\|x-\tilde x\\|\_{{\bf diag}(\tau)^{-1}}^2 \right)
    \end{array}
\\]
は \\({\bf R}^n \times \mathcal{K}^\ast \times \mathcal{K} \times {\bf R}\_+\\) への射影であり、
\\({\bf R}^n\\) への射影（これは何もしなくてよいということ）、\\({\bf R}\_+\\) への射影はやはり \\(\tau\\) に依存せず行うことができる。
また、仮に \\(\mathcal{K} = \mathcal{K}_1 \times \cdots \times \mathcal{K}_k\\) とした場合、
\\(\mathcal{K}^\ast = \mathcal{K}^\ast_1 \times \cdots \times \mathcal{K}^\ast_k\\) となり、
各 \\(\mathcal{K}_i,\ \mathcal{K}^\ast_i\\\) への射影どうしは独立している。
しかし \\(\mathcal{K}_i,\ \mathcal{K}^\ast_i\\\) への射影ひとつひとつは一般には \\(\tau\\) によるスケーリングに依存してしまう。

ここで、\\(\mathcal{K}_i\\) に対応する \\(\tau\\) の成分グループを \\(\tau\_{i1},\ldots,\tau\_{it}\\) と置く。
これらをすべて等しい値 \\(\tau\_i=\min(\tau\_{i1},\ldots,\tau\_{it})\\) に置き換えれば、等方スケールとなるので、
\\[
    {\bf prox}^{\bar \tau}\_{G}(\tilde x) =
    \Pi\_{{\bf R}^n \times \mathcal{K}^\ast \times \mathcal{K} \times {\bf R}\_+} (\tilde x)
\\]
と \\(\bar \tau\\) に依存しない形にすることができる
（\\(\mathcal{K}^\ast_i\\) も同様、\\(\tau\\) を全体的にグルーピングして置き換えたものを \\(\bar \tau\\) としている）。

なお、上記のグルーピングと置き換えにより
\\[
    \\|{\bf diag}(\sigma)^{\frac12} K {\bf diag}(\bar \tau)^{\frac12}\\|^2\le
    \\|{\bf diag}(\sigma)^{\frac12} K {\bf diag}(\tau)^{\frac12}\\|^2\le1
\\]
となり、収束条件は変わらず満たされていることを注記しておく。
