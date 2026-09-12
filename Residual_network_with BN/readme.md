# 11.4 批次正規化（Batch Normalization）

> 讀書筆記，整理自 *Understanding Deep Learning* (Simon J.D. Prince) 第 11.4 節。
> 內容為個人改寫摘要，公式與圖表概念版權屬原作者。

---

## 殘差網路中的變異數問題

參考原書 Figure 11.6：

| 情況 | 做法 | 結果 |
|------|------|------|
| (a) | 只用 He initialization | 每個殘差區塊的輸入會被加回輸出，**變異數每層加倍**（1 → 2 → 4 → 8），呈指數成長 |
| (b) | 每個殘差區塊之間乘上 $`\frac{1}{\sqrt{2}}`$ | 把訊號重新縮放回去，變異數維持穩定 |
| (c) | 把 BN 放在殘差區塊的第一步，並令 $`\delta = 0`$、$`\gamma = 1`$ | 每層輸入變成單位變異數，變異數隨區塊數**線性**成長（1 → 2 → 3 → 4） |

> 方法 (c) 的副作用：初始化時後面的層被殘差連結主導，因此接近在計算恆等映射（identity）。

<!-- 若要放原書截圖，把圖片放進 images/ 資料夾後取消下一行註解：
![Figure 11.6 — Variance in residual networks](images/fig11-6.png)
-->

---

## 公式

對一個 batch $`\mathcal{B}`$，先計算該批次的統計量（式 11.7）：

$$
m_h = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} h_i
$$

$$
s_h = \sqrt{\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} (h_i - m_h)^2}
$$

接著把批次的激活值標準化成平均 0、變異數 1（式 11.8）：

$$
h_i \leftarrow \frac{h_i - m_h}{s_h + \epsilon}, \quad \forall i \in \mathcal{B}
$$

其中 $`\epsilon`$ 是很小的常數，避免當 batch 內所有 $`h_i`$ 都相同（$`s_h = 0`$）時除以零。

最後用可學習參數 $`\gamma`$ 縮放、$`\delta`$ 平移（式 11.9）：

$$
h_i \leftarrow \gamma h_i + \delta, \quad \forall i \in \mathcal{B}
$$

---

## 11.4.1 批次正規化的代價與好處

### 代價

- BN 讓網路對權重與偏差的縮放**不敏感**：權重加倍 → 激活值加倍 → $`s_h`$ 也加倍 → 式 11.8 剛好抵銷。因此存在一大族群的權重組合會產生完全相同的效果。
- 每個隱藏單元多了 $`\gamma`$、$`\delta`$ 兩個參數，模型變大。
- 結論：BN 同時製造了**冗餘**（權重）又**新增參數**來補償那個冗餘，效率上並不漂亮 —— 但好處足以彌補。

### 好處

#### 1. 前向傳遞穩定（Stable forward propagation）

初始化時設 $`\delta = 0`$、$`\gamma = 1`$，每個隱藏單元的輸出都有單位變異數。

- **一般網路**：變異數在初始化時全程穩定。
- **殘差網路**：變異數仍會隨層數增加，但變成**線性**成長 —— 第 $`k`$ 層加上一單位變異數。
- **副作用**：初始化時後面的層對整體變化的貢獻較小，網路「有效深度」比實際淺；訓練過程中網路可以自己調大後層的 $`\gamma`$ 來控制有效深度。

#### 2. 可用更大的學習率（Higher learning rates）

實驗與理論都指出 BN 讓 loss surface 與其梯度變化更平滑（梯度較不分散），surface 更可預測，因此能用更大的學習率 —— 而更大的學習率通常能改善測試表現（見原書 9.2 節）。

#### 3. 正規化效果（Regularization）

訓練過程中的雜訊有助於泛化（見原書第 9 章）。BN 因為依賴 batch 統計量而**注入雜訊**：同一筆訓練樣本的激活值，會依照同 batch 內其他成員而被標準化成不同的量，每次迭代都不一樣。

---

## 參考資料

- Simon J.D. Prince, *Understanding Deep Learning*, MIT Press — §11.4
- Ioffe & Szegedy (2015), *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*
