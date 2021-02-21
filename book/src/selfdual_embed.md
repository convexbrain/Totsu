# Homogeneous Self-dual Embedding

[錐線形計画問題の最適性条件](./conic_lp.md#最適性条件)を線形方程式系として書くと
\\[
    \left[ \begin{matrix}
    0 & A^T & 0 \\\\
    -A & 0 & -I \\\\
    c^T & b^T & 0
    \end{matrix} \right]
    \left[ \begin{matrix}
    x \\\\ y \\\\ s
    \end{matrix} \right]
    +
    \left[ \begin{matrix}
    c \\\\ b \\\\ 0
    \end{matrix} \right]
    = 0, \qquad
    s \in \mathcal{K}, \quad
    y \in \mathcal{K}^*
\\]
となるが、主問題・双対問題が実行可能でない場合には解をもたない。
そこで Homogeneous Self-dual Embedding（[参考文献](./reference.md)[1]を参照）に倣い、新たな変数 \\(\tau,\kappa\\) を導入して、
\\[
    \left[ \begin{matrix}
    0 & A^T & 0 & c\\\\
    -A & 0 & -I & b \\\\
    -c^T & -b^T & 0 & 0
    \end{matrix} \right]
    \left[ \begin{matrix}
    x \\\\ y \\\\ s \\\\ \tau
    \end{matrix} \right]
    =
    \left[ \begin{matrix}
    0 \\\\ 0 \\\\ \kappa
    \end{matrix} \right]
    , \qquad
    s \in \mathcal{K}, \quad
    y \in \mathcal{K}^*, \quad
    \tau \in {\bf R}\_+, \quad
    \kappa \in {\bf R}\_+
\\]
とおく。
特に \\(\tau=1,\kappa=0\\) の場合は元の線形方程式に戻る。
また
\\[
    \begin{array}{ll}
    &-c^Tx -b^Ty = \kappa \\\\
    \Longrightarrow & -\tau c^Tx -\tau b^Ty = \tau\kappa \\\\
    \Longrightarrow & (A^Ty)^Tx -(Ax+s)^Ty = \tau\kappa \\\\
    \Longrightarrow & -s^Ty = \tau\kappa \\\\
    \end{array}
\\]
と計算でき、双対錐の定義から \\(s^Ty\ge0\\) なので \\(\tau\kappa\le0\\) となり、
\\(\tau,\kappa \in {\bf R}\_+\\) から少なくともいずれかは \\(0\\) となる。

## \\(\tau>0,\kappa=0\\) の場合
\\(x/\tau,y/\tau,s/\tau\\) が最適性条件を満たすことがわかるので、これらは錐線形計画問題（主問題・双対問題）の解となる。

## \\(\tau=0,\kappa>0\\) の場合

主問題または双対問題が実行可能解をもたない。
[参考文献](./reference.md)[1]を参照。

## \\(\tau=0,\kappa=0\\) の場合

実行可能解をもたないか、あるいは解について何も言えることがない状況である。
[参考文献](./reference.md)[1]を参照[^totsu1]。

---

[^totsu1]: 自明な解（オールゼロ）に向かわないことが証明されている。`Totsu`では、[Pock/Chambolleの一次法](./pock_chambolle.md)を適用しているが、この場合にも自明な解に向かわないことを示す必要があり、今後の課題のひとつ。
