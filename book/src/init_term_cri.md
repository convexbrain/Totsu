# 初期値と終了基準

初期値は
\\[
    x_0=
    \left[ \begin{matrix}
    \hat x_0 \\\\ \hat y_0 \\\\ \hat s_0 \\\\ \hat \tau_0
    \end{matrix} \right]
    =
    \left[ \begin{matrix}
    0 \\\\ 0 \\\\ 0 \\\\ 1
    \end{matrix} \right]
    ,\qquad
    y_0=0
\\]
とする。

[Pock/Chambolleの一次法](./pock_chambolle.md)の反復において、\\(x_k\\) は \\({\bf prox}^\tau\_{G}\\) による射影の後のため必ず
\\(x_k \in {\bf R}^n \times \mathcal{K}^\ast \times \mathcal{K} \times {\bf R}\_+\\) となっており、
主・双対の錐の制約条件を満たしている。
この状況のもと、終了基準は[参考文献](./reference.md)[1]と同等のものを適用する。

## \\(\hat \tau^k > \varepsilon_{\rm zero}\\) の場合

### 収束判定

\\[
    \begin{array}{l}
    & \frac{\\|A \hat x_k / \hat\tau_k + \hat s_k / \hat\tau_k - b\\|}{1 + \\|b\\|} \le \varepsilon_{\rm acc} \\\\
    \land& \frac{\\|A^T \hat y_k / \hat\tau_k + c\\|}{1 + \\|c\\|} \le \varepsilon_{\rm acc} \\\\
    \land& \frac{\\|c^T \hat x_k / \hat\tau_k + b^T \hat y_k / \hat\tau_k\\|}{1 + |c^T \hat x_k / \hat\tau_k| + |b^T \hat y_k / \hat\tau_k|} \le \varepsilon_{\rm acc}
    \end{array}
\\]
を満たすとき、解 \\(x^\star=\hat x_k / \hat\tau_k,\ y^\star=\hat y_k / \hat\tau_k\\) に収束したと判定し、終了する。

## \\(\hat \tau^k \not > \varepsilon_{\rm zero}\\) の場合

### 下限なし判定

\\[
    -c^T \hat x_k > \varepsilon_{\rm zero}
    \quad \land \quad
    \\|A \hat x_k + \hat s_k\\|\frac{\\|c\\|}{-c^T \hat x_k} \le \varepsilon_{\rm inf}
\\]
を満たすとき、主問題の下限なし（双対問題の実行可能な解なし）と判定し、終了する。

### 実行不可能判定

\\[
    -b^T \hat y_k > \varepsilon_{\rm zero}
    \quad \land \quad
    \\|A^T \hat y_k\\|\frac{\\|b\\|}{-b^T \hat y_k} \le \varepsilon_{\rm inf}
\\]
を満たすとき、主問題の実行可能な解なしと判定し、終了する。
