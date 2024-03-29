# 線形合成項を含み、滑らかでない関数からなる、分離可能な凸最適化問題

## 主問題

$$
    \begin{array}{ll}
    \mathrm{minimize} & G(x) + F(Kx)
    \end{array}
$$
を考える。ここで、
* 変数 $x\in\mathbb{R}^n$
* $K\in\mathbb{R}^{m\times n}$
* $G: \mathbb{R}^n\to\mathbb{R}\cup\lbrace+\infty\rbrace,\ F: \mathbb{R}^m\to\mathbb{R}\cup\lbrace+\infty\rbrace$
  は下半連続な凸関数
  * このとき $G^{\ast\ast}=G,\ F^{\ast\ast}=F$ となる
    * ただし $h^\ast(y)=\sup_x(y^Tx-h(x))$ は $h$ の共役関数

である。

$F(Kx)$ という形で線形合成項を含み、$G,F$ は下半連続であれば滑らかである必要はなく、
目的関数が2項の和に分離可能な凸最適化問題である。

## 双対問題

新たな変数 $z\in\mathbb{R}^m$ を介して主問題を
$$
    \begin{array}{ll}
    \mathrm{minimize} & G(x) + F(z) \\
    \mathrm{subject\ to} & Kx = z
    \end{array}
$$
と書き直し、双対変数あるいはラグランジュ乗数を $y\in\mathbb{R}^m$ としてラグランジアン
$$
    L = G(x) + F(z) + y^T(Kx - z)
$$
を導入する。

$$
    \begin{array}{ll}
    & \inf_{x,z} L \\
    = & \inf_x(G(x) + y^TKx) + \inf_z(F(z) - y^Tz) \\
    = & - \sup_x((-K^Ty)^Tx - G(x)) - \sup_z(y^Tz - F(z)) \\
    = & - G^\ast(-K^Ty) - F^\ast(y)
    \end{array}
$$
より、これを最大化する双対問題
$$
    \begin{array}{ll}
    \mathrm{maximize} & -(G^\ast(-K^Ty) + F^\ast(y))
    \end{array}
$$
が得られる。

## 鞍点問題

なお、$L_z=\inf_z L$ とおくと
$$
    L_z = (Kx)^Ty + G(x) - F^\ast(y)
$$
となり、双対問題は $\max_y\inf_xL_z$ とも書ける。
一方
$$
    \begin{array}{ll}
    & \sup_y L_z \\
    = & G(x) + \sup_y((Kx)^Ty - F^\ast(y)) \\
    = & G(x) + F^{\ast\ast}(Kx)
    \end{array}
$$
より、主問題は $\min_x\sup_yL_z$ と表すことができる。

したがって
$$
    \begin{array}{l}
    \min_x \max_y & (Kx)^Ty + G(x) - F^\ast(y)
    \end{array}
$$
は、上記主・双対問題の鞍点問題を定めている。
