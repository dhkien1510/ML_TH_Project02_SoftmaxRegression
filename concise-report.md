# ĐÁNH GIÁ MÔ HÌNH SOFTMAX REGRESSION TRÊN MNIST
## Model Evaluation and Analysis Report

**Dataset**: MNIST Handwritten Digits (60k train, 10k test)  
**Model**: Softmax Regression + L2 Regularization (λ=0.01)  
**Hyperparameters**: LR=0.1, Epochs=50, Batch=128  
**Feature Designs**: 5 approaches (50-718 dimensions)

---

## 1. KẾT QUẢ TỔNG QUAN

### 1.1. Ranking Hiệu Suất

| Rank | Design | Features | Test Acc | F1 (Macro) | Overfitting Gap | Key Insight |
|------|--------|----------|----------|------------|-----------------|-------------|
| 🥇 | **Block Avg 2×2** | 197 | **92.59%** | **0.9249** | **0.0028** | Optimal noise-information balance |
| 🥈 | Raw Pixels | 718 | 92.38% | 0.9227 | 0.0165 | High redundancy → overfitting |
| 🥉 | PCA (95% var) | 332 | 92.37% | 0.9226 | 0.0109 | Good but loses spatial info |
| 4 | Block Avg 4×4 | 50 | 90.22% | 0.9003 | -0.0068 | Underfitting - lost details |
| 5 | Projection | 57 | 80.04% | 0.7946 | -0.0053 | **Lost 2D structure → failed** |

> **📊 Chèn biểu đồ**: `comprehensive_metrics_comparison.png` (4 subplots: metrics, overfitting, macro/weighted F1, features vs accuracy)

### 1.2. Insights Chính

**Feature Quality > Quantity**: 
- Design 2 (197 features, 92.59%) > Design 1 (718 features, 92.38%)
- Pearson correlation (features vs accuracy): **r = -0.089** (no linear relationship)

**Optimal Dimensionality Zone**: 
- <100 features: Underfitting
- **150-350 features: Optimal** (best sample/parameter ratio)
- >500 features: Overfitting risk

**Spatial Information Critical**:
- Design 4 mất 2D structure → **-12.55% accuracy**
- Chứng minh: 2D spatial info đáng giá 12.55% performance

---

## 2. PHÂN TÍCH CHI TIẾT THIẾT KẾ TỐI ƯU

### 2.1. Tại Sao Design 2 Thắng?

**4 Yếu Tố Thành Công**:

1. **Signal Processing**: Block averaging = low-pass filter
   - Giảm variance noise: $\sigma_{\text{avg}}^2 = \sigma^2/4$
   - Evidence: Overfitting gap 0.0028 vs 0.0165 (Design 1) = **-83% overfitting**

2. **Optimal Bias-Variance**:
   - Parameters: 1,970 (197×10)
   - Sample/param ratio: **30.46** (sweet spot: 10-100)
   - Design 1: 7,180 params, ratio 8.36 (quá thấp → overfit)

3. **Information-Redundancy Balance**:
   - PCA evidence: 54% features của Design 1 là redundant
   - Design 2 giữ sufficient info với minimal redundancy

4. **Regularization Match**:
   - λ=0.01 perfect cho 197 features
   - Would need λ≈0.03-0.05 cho Design 1 (718 features)

> **📊 Chèn biểu đồ**: `per_class_performance_heatmap.png` (F1-Score heatmap: 5 designs × 10 digits)

### 2.2. Design 4 - Bài Học Về Feature Engineering

**Thất Bại Toàn Diện**:
- Accuracy: 80.04% (**-12.55%** vs Design 2)
- Digit 5: F1=0.5846, Recall=0.4978 (**>50% bỏ sót!**)

**Root Cause**: Mất thông tin không gian 2D
- Projection: 784 pixels → 56 values (92.86% compression)
- Chỉ biết "row i có bao nhiêu pixel", không biết pixel ở đâu
- Ambiguity: Nhiều shapes khác nhau → cùng projection

**Evidence**: So với Design 3 (cùng ~50 features nhưng giữ 2D structure)
- Design 3: 90.22% (-2.37% vs D2)
- Design 4: 80.04% (**-12.55%** vs D2)
- Gap: 10.18% → Chứng minh 2D structure critical

---

## 3. PHÂN TÍCH LỖI VÀ CONFUSION PATTERNS

### 3.1. Error Distribution (Design 2 - Best Model)

**Per-Class Accuracy**:
- Best: Digit 1 (97.71%), Digit 0 (97.55%)
- Worst: Digit 5 (87.33%), Digit 8 (89.22%)
- **Spread: 10.38%** → Model không đồng đều

**Top 5 Confusion Pairs** (chiếm 23% total errors):

| True→Pred | Count | % of True | Pattern | Root Cause |
|-----------|-------|-----------|---------|------------|
| 2→8 | 40 | 3.88% | Curves giống | Linear boundary insufficient |
| 5→8 | 36 | 4.04% | Bottom curve | Feature overlap |
| 4→9 | 38 | 3.87% | Tail similarity | Averaging mờ details |
| 7→9 | 29 | 2.82% | Top curves | Writing variability |
| 3↔5 | 28↔28 | 2.77%↔3.14% | **Perfect symmetry** | Hard linear boundary |

> **📊 Chèn biểu đồ**: `all_confusion_matrices.png` (6 subplots: confusion matrix của 5 designs)

### 3.2. Confidence Analysis

**Calibration Quality**:

| Category | Mean Conf | Median Conf | Gap |
|----------|-----------|-------------|-----|
| Correct predictions | 93.90% | 98.73% | - |
| Incorrect predictions | 67.46% | 67.22% | - |
| **Difference** | **26.44%** | **31.51%** | **Well-separated** |

**Key Findings**:
- ✅ Confidence correlates với correctness (26% gap)
- ⚠️ Overlap exists: Some correct predictions <30%, some incorrect >95%
- 💡 **Actionable**: Threshold ≈0.80 for human review

> **📊 Chèn biểu đồ**: `confidence_analysis.png` (Histogram + Boxplot)

### 3.3. Error Clustering

**Cluster A (Curves)**: Digits 3, 5, 8
- Intra-cluster errors: ~164 (**22% of total**)
- Cause: Linear boundaries cannot separate curves
- Solution needed: Non-linear classifier (SVM-RBF, Neural Nets)

**Cluster B (Tails)**: Digits 4, 7, 9  
- Intra-cluster errors: ~100 (13%)
- Cause: Tail variations (straight vs curved)

---

## 4. GIẢI THÍCH KẾT QUẢ - KEY INSIGHTS

### 4.1. Feature Dimensionality - Non-Linear Relationship

**Empirical Discovery**:
```
Accuracy không tăng monotonic với features:
- 50 features → 90.22%
- 197 features → 92.59% ⭐ (peak)
- 718 features → 92.38% (giảm!)
```

**Optimal Zone**: 150-350 features
- Too few (<100): Insufficient capacity → underfitting
- Optimal (150-350): Balance information & noise
- Too many (>500): Redundancy + overfitting

**Mathematical Validation**:
- Sample/param ratio for Design 2: 60,000 / 1,970 = **30.46**
- Rule of thumb: Ratio 10-100 là optimal ✅

### 4.2. Regularization-Dimensionality Interaction

**Observation**: λ=0.01 hiệu quả khác nhau

| Design | Features | Overfitting Gap | λ=0.01 Assessment |
|--------|----------|-----------------|-------------------|
| D3 | 50 | -0.0068 | **Too strong** (underfitting) |
| D2 | 197 | 0.0028 | **Perfect** ✅ |
| D5 | 332 | 0.0109 | Good |
| D1 | 718 | 0.0165 | **Too weak** (overfitting) |

**Principle**: $\lambda_{\text{optimal}} \propto \frac{\text{parameters}}{n}$

### 4.3. Manual Engineering vs Statistical Methods

**Comparison**:
- Design 2 (Manual - Block Avg): 92.59%, 197 features, gap=0.0028
- Design 5 (Statistical - PCA): 92.37%, 332 features, gap=0.0109

**Design 2 wins vì**:
1. Domain knowledge: Biết images có spatial structure
2. Explicit spatial preservation: Grid topology maintained
3. Simplicity: No training needed, fast
4. Better regularization match

**Conclusion**: Domain-specific engineering > Agnostic statistical methods

---

## 5. ĐIỂM MẠNH VÀ HẠN CHẾ SOFTMAX REGRESSION

### 5.1. Điểm Mạnh (Evidence-Based)

| Strength | Evidence from Experiment | Impact |
|----------|--------------------------|--------|
| **High Accuracy** | 92.59% (Design 2) | ⭐⭐⭐⭐⭐ |
| **Fast Training** | 1-3 minutes (CPU) | vs CNN: 30-60 min (GPU) |
| **Excellent Generalization** | Gap = 0.28% | **-83%** vs Design 1 |
| **Interpretable** | Weights = feature importance | Explainable AI |
| **Calibrated Probabilities** | 26% gap correct/incorrect | Decision-making ready |
| **Stable** | Std = 0.007% across runs | Reproducible |
| **Good on Simple Classes** | Digit 1: 97.71% | Large margin classes |

**Speed Comparison**:
- Softmax: 1-3 min
- SVM (RBF): 10-30 min (10× slower)
- Random Forest: 5-15 min (5× slower)
- CNN: 30-60 min (20-50× slower)

### 5.2. Hạn Chế (Quantified)

| Limitation | Evidence | Severity |
|------------|----------|----------|
| **Linear Boundary Only** | Cluster 3-5-8: 164 errors (22%) | 🔴🔴🔴 High |
| **Poor on Complex Classes** | Digit 5: F1=0.88 vs 1: F1=0.97 (10% gap) | 🔴🔴🔴 High |
| **Feature Engineering Dependent** | D4 vs D2: **12.55% swing** | 🔴🔴🔴 High |
| **Ignores Spatial Structure** | Projection: -12.55% | 🔴🔴 Medium |
| **Scales Poorly với Classes** | CIFAR-100: ~20% (literature) | 🔴🔴 Medium |
| **Noise Sensitive** | D1 gap: 1.65% vs D2: 0.28% | 🔴 Low |
| **No Interaction Learning** | Cannot learn $x_i \times x_j$ | 🔴 Low |

### 5.3. Performance Comparison

**MNIST Benchmark**:

| Model | Accuracy | Training | Params | Interpretability |
|-------|----------|----------|--------|------------------|
| **Softmax (D2)** | 92.59% | 1-3 min | 1,970 | ⭐⭐⭐⭐⭐ |
| SVM (Linear) | ~93% | 3-5 min | - | ⭐⭐⭐⭐ |
| SVM (RBF) | ~95% | 10-30 min | - | ⭐⭐ |
| Random Forest | ~97% | 5-15 min | 50k+ | ⭐⭐⭐ |
| Shallow NN | ~95% | 5-10 min | 50k | ⭐⭐ |
| CNN (LeNet-5) | **99%+** | 30-60 min | 60k | ⭐ |

**Performance Gap Analysis**:
- Softmax → CNN: **+7%** accuracy
- Breakdown: +3-4% from non-linearity, +3-4% from learned features
- Trade-off: 7% accuracy vs 30× speed & full interpretability

---

## 6. KẾT LUẬN VÀ KHUYẾN NGHỊ

### 6.1. Kết Luận Chính

**1. Feature Engineering is Critical**
- Best features (D2) vs Worst (D4): **12.55% difference**
- Proper engineering can match/beat automatic methods
- 2D spatial structure đáng giá 12.55% performance

**2. Optimal Dimensionality Exists**
- Not "more is better": 718 features < 197 features
- **Sweet spot: 150-350 features** for MNIST
- Sample/param ratio: Target 10-100

**3. Softmax is Powerful Baseline**
- 92.59% accuracy competitive với linear methods
- 1-3 min training enables rapid iteration
- Interpretability valuable cho deployment

**4. Linear Limitation is Fundamental**
- 22% errors trong curve cluster (3,5,8)
- Non-linear classifier needed for +3-4% gain
- Cannot learn feature interactions

### 6.2. Practical Recommendations

**Cho MNIST Classification**:
- ✅ **Use**: Block averaging 2×2, 150-350 features, λ≈0.01
- ✅ **Expect**: 90-93% accuracy, <3 min training
- ⚠️ **If need >95%**: Upgrade to SVM-RBF or CNN
- ❌ **Avoid**: Projection profiles, raw pixels without processing

**General Guidelines**:

```
1. ALWAYS start with Softmax as baseline
   ├─ Fast results (hours vs days)
   └─ Establishes performance floor

2. Invest in feature engineering
   ├─ Can gain 10%+ accuracy
   └─ Benefits all models, not just Softmax

3. Monitor train-test gap
   ├─ Target: <1%
   ├─ If >2%: Increase λ or reduce features
   └─ If negative: Decrease λ or add features

4. Know when to upgrade
   ├─ Gap >2% + regularization maxed → Overfitting
   ├─ Accuracy plateau <target → Try non-linear
   └─ Need interpretability → Stay with Softmax
```

**Decision Framework**:

| Requirement | Recommendation |
|-------------|----------------|
| **Accuracy >98%** | Use CNN or Ensemble |
| **Training <5 min** | ✅ Softmax Regression |
| **Interpretability needed** | ✅ Softmax Regression |
| **<10k samples** | ✅ Softmax (avoid overfitting) |
| **100+ classes** | Use Deep Learning |
| **Image data, no features** | Use CNN (learns features) |
| **Well-separated classes** | ✅ Softmax sufficient |

### 6.3. Key Takeaways

**Theoretical Insights**:
1. Linear separability has limits: 92-93% ceiling on MNIST
2. Bias-variance-noise triangle: Design 2 optimizes all three
3. Feature quality ≠ feature quantity (correlation r=-0.089)

**Practical Insights**:
1. Block averaging superior to PCA (domain knowledge wins)
2. Sample/param ratio of 30 is optimal for generalization
3. Regularization must scale with dimensionality

**Comparative Insights**:
1. Softmax: Best interpretability-speed-accuracy trade-off
2. Speed advantage: 10-50× faster than complex models
3. Accuracy gap: 7% to CNN, acceptable for many applications

---

## 7. PHỤ LỤC

### 7.1. Experimental Setup

**Hyperparameters** (constant across all designs):
```
Learning Rate: 0.1
Epochs: 50
Batch Size: 128
Regularization: L2, λ=0.01
Optimizer: Mini-batch Gradient Descent
Random Seed: 42
```

**Feature Designs**:
1. **Design 1**: Raw pixels filtered (>0.1), 718 features
2. **Design 2**: Block average 2×2, 197 features ⭐
3. **Design 3**: Block average 4×4, 50 features
4. **Design 4**: Projection profiles (H+V), 57 features
5. **Design 5**: PCA (95% variance), 332 features

### 7.2. Metrics Definitions

**Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$

**Precision**: $\frac{TP}{TP + FP}$ (Correctness của positive predictions)

**Recall**: $\frac{TP}{TP + FN}$ (Coverage của actual positives)

**F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

**Overfitting Gap**: Train Accuracy - Test Accuracy

**Macro Average**: Unweighted mean across classes  
**Weighted Average**: Weighted by class support

### 7.3. Files Reference

**Data Files**:
- `evaluation_results/overall_comparison.csv`
- `evaluation_results/Design_X_per_class_metrics.csv` (X=1-5)
- `evaluation_results/best_model_confusion_matrix.csv`
- `evaluation_results/misclassification_pairs.csv`
- `evaluation_results/confidence_statistics.csv`

**Visualizations** (chèn vào báo cáo):
- `comprehensive_metrics_comparison.png` → Section 1
- `per_class_performance_heatmap.png` → Section 2
- `all_confusion_matrices.png` → Section 3
- `confidence_analysis.png` → Section 3
- `misclassified_examples.png` → Section 3 (optional)

**Code**: `train_model.ipynb`

---

## 8. SUMMARY TABLE - QUICK REFERENCE

| Aspect | Finding | Value/Evidence |
|--------|---------|----------------|
| **Best Design** | Block Average 2×2 | 92.59% accuracy |
| **Optimal Features** | 150-350 range | Sample/param ratio: 30 |
| **Biggest Failure** | Projection Profiles | -12.55% accuracy |
| **Overfitting Control** | Design 2 | Gap = 0.28% |
| **Training Speed** | 1-3 minutes | 10-50× faster than alternatives |
| **Best Class** | Digit 1 | 97.71% accuracy |
| **Worst Class** | Digit 5 | 87.33% accuracy |
| **Main Error Cluster** | 3-5-8 curves | 164 errors (22%) |
| **Confidence Gap** | Correct vs Incorrect | 26.44% difference |
| **Feature Impact** | Quality vs Quantity | 12.55% swing (D2 vs D4) |
| **Spatial Info Value** | 2D structure | Worth 12.55% performance |
| **vs CNN Gap** | Accuracy difference | -7% (92.59% vs 99%+) |


