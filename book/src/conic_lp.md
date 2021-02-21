# 錐線形計画問題

錐線形計画問題は、凸最適化問題のうち錐最適化問題に属するクラスの問題である。
一見線形計画とあるので扱える問題が狭そうに思えるが、実際は典型的な非線形計画問題を含んでいる：

* 凸最適化問題 \\(\supset\\) 錐最適化問題 \\(\supset\\) 錐線形計画問題 \\(\supset\\)
  SDP \\(\supset\\) SOCP \\(\supset\\) QCQP \\(\supset\\) QP \\(\supset\\) LP

上記SDP～QPにおける非線形な目的関数・制約条件は、錐線形計画問題においては凸錐の制約条件によって表される。

錐は以下のように特徴づけられる集合である：
ある錐に属する任意の元 \\(x\\) と、任意のスカラー \\(\lambda>0\\) について、\\(\lambda x\\) もまたその錐に含まれる。

## 主問題

\\[
    \begin{array}{l}
    {\rm minimize} & c^Tx \\\\
    {\rm subject\ to} & Ax \preceq_\mathcal{K} b
    \end{array}
\\]
ここで、
* 変数 \\(x\in{\bf R}^n\\)
* \\(c\in{\bf R}^n,\ A\in{\bf R}^{m\times n},\ b\in{\bf R}^m\\)
* 閉凸錐 \\(\mathcal{K}\ne\emptyset\\)

とする。
また \\(\preceq_\mathcal{K}\\) の関係は
\\[
    x\preceq_\mathcal{K}y \Longleftrightarrow
    0\preceq_\mathcal{K}y-x \Longleftrightarrow
    y-x\in\mathcal{K}
\\]
であり、スラック変数 \\(s\in{\bf R}^m\\) を導入して主問題は
\\[
    \begin{array}{l}
    {\rm minimize} & c^Tx \\\\
    {\rm subject\ to} & Ax + s = b \\\\
    & s \in \mathcal{K}
    \end{array}
\\]
と書くことができる。

## 双対問題

双対変数あるいはラグランジュ乗数を \\(y\\) とし、
ラグランジアン
\\[
    L(x,s;y) = c^Tx + y^T(Ax+s-b)
\\]
を導入する。
ここで注意として、\\(x,y\\) の定義域はそれぞれ \\({\bf R}^n, {\bf R}^m\\) だが、
\\(s\\) の定義域は \\(\mathcal{K}\\) としている。

ラグランジュ双対をとるために \\(\inf_{x,s\in\mathcal{K}} L\\) を評価するにあたり、
\\[
    L = (c + A^Ty)^Tx + y^Ts - b^Ty
\\]
から、
* \\(c + A^Ty \ne 0\\) のとき、適当な \\(x\\) で \\(L\\) はいくらでも小さくできる
* ある \\(y,s\\) で \\(y^Ts<0\\) となるとき、
  \\(\mathcal{K}\\) は錐なので任意の \\(\lambda>0\\) に対して \\(\lambda s\\) も \\(\mathcal{K}\\) に含まれ、
  \\(y^T\lambda s\\) は（よって \\(L\\) も）いくらでも小さくできる

ことがわかる。

したがって \\(\inf_{x,s\in\mathcal{K}} L > -\infty\\) のためには
* \\(c + A^Ty = 0\\)
* \\(y^Ts \ge 0,\ \forall s\in\mathcal{K}\\)
  * このような \\(y\\) からなる集合は双対錐の定義と一致し、\\(y\in\mathcal{K}^*\\) と書く
  * なお \\(\mathcal{K}\\) は閉集合としたので \\(0\in\mathcal{K}\\)、よって \\(y^Ts\\) は必ず \\(0\\) を含む

となる必要があり、まとめると
\\[
    \inf_{x,s\in\mathcal{K}} L =
    \left\lbrace \begin{array}{l}
    -b^Ty &  ({\rm if}\ c + A^Ty = 0,\ y\in\mathcal{K}^*) \\\\
    -\infty & ({\rm otherwise})
    \end{array} \right.
\\]

となる。

結果として \\(g(y)=\inf_{x,s\in\mathcal{K}} L\\) を最大化するラグランジュ双対問題をとると
\\[
    \begin{array}{l}
    {\rm maximize} & -b^Ty \\\\
    {\rm subject\ to} & -A^Ty = c \\\\
    & y \in \mathcal{K}^*
    \end{array}
\\]
が得られる。

なお逆に \\(\sup_{y} L\\)（注：\\(y\\) の定義域 \\({\bf R}^m\\) での \\(\sup\\)）の最小化が主問題に戻ることがわかる。

## 弱双対性

\\(s\\) の定義域 \\(\mathcal{K}\\) に注意して
\\[
    L(x,s;y) \ge \inf_{x,s\in\mathcal{K}} L = g(y)
\\]
であるから、
\\(\hat x, \hat s, \hat y\\) を主問題・双対問題の実行可能解のひとつとすると、
\\[
    c^T \hat x = L(\hat x, \hat s; \hat y) \ge g(\hat y) = -b^T \hat y
\\]
となる。
つまり、実行可能領域において双対問題は主問題の下限を与えている。

さらに最適解 \\(x^\star,s^\star,y^\star\\) を代入すれば、最適値を \\(p^\star,d^\star\\) として
\\[
    p^\star = c^T x^\star \ge -b^T y^\star = d^\star
\\]
が成立する。

## 強双対性

錐線形計画問題は凸最適化問題の一種であり、制約想定（スレーターの条件など）のもとで強双対性が成立し、
双対ギャップがゼロ、つまり主問題と双対問題の最適値が一致する。
\\[
    p^\star = d^\star
\\]
[参考文献](./reference.md)[5]を参照。

## 最適性条件

強双対性の仮定のもと、
\\[
    Ax + s = b,\quad
    s \in \mathcal{K},\quad
    -A^Ty = c,\quad
    y \in \mathcal{K}^*,\quad
    c^Tx = -b^Ty
\\]
が最適性の必要十分条件（KKT条件、[参考文献](./reference.md)[5]を参照）となることがわかる。
