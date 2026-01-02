 
برای اینکه **جزییات کامل دیتاست Iris** را استخراج کنیم، معمولاً چند نوع اطلاعات مهم را بررسی می‌کنیم:

- شکل داده‌ها (Dimensions)  
- نام ویژگی‌ها  
- نوع داده‌ها  
- آمار توصیفی (Descriptive Statistics)  
- تعداد کلاس‌ها و توزیع آن‌ها  
- نمونه‌هایی از داده  

در ادامه یک کد کامل و خروجی‌های مورد انتظار را بهت می‌دهم.

---

# 🔍 کد استخراج کامل جزییات دیتاست Iris

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# 1. Shape of dataset
print("Shape:", df.shape)

# 2. Feature names
print("\nFeatures:", iris.feature_names)

# 3. Target names
print("\nTarget classes:", iris.target_names)

# 4. Data types
print("\nData types:\n", df.dtypes)

# 5. Descriptive statistics
print("\nDescriptive statistics:\n", df.describe())

# 6. Class distribution
print("\nClass distribution:\n", df['species'].value_counts())

# 7. First 5 rows
print("\nSample rows:\n", df.head())
```

---

# 📊 خروجی و تحلیل هر بخش

## 1) **Shape**
```
(150, 5)
```
یعنی:
- 150 نمونه  
- 4 ویژگی + 1 ستون کلاس  

---

## 2) **Features**
```
['sepal length (cm)', 'sepal width (cm)', 
 'petal length (cm)', 'petal width (cm)']
```

---

## 3) **Target classes**
```
['setosa' 'versicolor' 'virginica']
```

---

## 4) **Data types**
```
sepal length (cm)    float64
sepal width (cm)     float64
petal length (cm)    float64
petal width (cm)     float64
species                int64
```

---

## 5) **Descriptive statistics**
نمونه‌ای از خروجی:

| Feature | mean | std | min | max |
|--------|------|------|------|------|
| sepal length | 5.84 | 0.83 | 4.3 | 7.9 |
| sepal width | 3.05 | 0.43 | 2.0 | 4.4 |
| petal length | 3.76 | 1.76 | 1.0 | 6.9 |
| petal width | 1.20 | 0.76 | 0.1 | 2.5 |

---

## 6) **Class distribution**
```
0 (setosa)        50
1 (versicolor)    50
2 (virginica)     50
```

کاملاً **متعادل** است.

---

## 7) **نمونه‌ای از داده‌ها**
```
   sepal length  sepal width  petal length  petal width  species
0           5.1          3.5           1.4          0.2        0
1           4.9          3.0           1.4          0.2        0
2           4.7          3.2           1.3          0.2        0
3           4.6          3.1           1.5          0.2        0
4           5.0          3.6           1.4          0.2        0
```

---

# 🌱 جمع‌بندی

با این کد و خروجی‌ها، تو:

- ساختار دیتاست  
- آمار توصیفی  
- توزیع کلاس‌ها  
- نوع داده‌ها  
- نمونه‌های اولیه  

را کاملاً استخراج کردی.

---
