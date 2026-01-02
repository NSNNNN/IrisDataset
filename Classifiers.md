بیایید روی دیتاست **Iris** چندین **Classifier** مختلف را امتحان کنیم، آن‌ها را آموزش بدهیم و دقت‌شان را با هم مقایسه کنیم.  
این دقیقاً همان کاری است که در پروژه‌های واقعی ML انجام می‌دهیم.

در این مثال از ۵ مدل استفاده می‌کنیم:

1. **Logistic Regression**  
2. **Decision Tree**  
3. **Random Forest**  
4. **SVM (Support Vector Machine)**  
5. **Naive Bayes**

و سپس نتایج را کنار هم قرار می‌دهیم.

---

# 🔥 کد کامل مقایسهٔ Classifierها روی دیتاست Iris

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

# Train & evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

results
```

---

# 📊 نتایج احتمالی (تقریباً مشابه اجراهای واقعی)

| مدل | دقت (Accuracy) |
|------|----------------|
| Logistic Regression | 1.00 |
| Decision Tree | 1.00 |
| Random Forest | 1.00 |
| SVM | 1.00 |
| Naive Bayes | 0.96 |

> توجه: چون دیتاست Iris ساده و تمیز است، اکثر مدل‌ها دقت بسیار بالا می‌گیرند.

---

# 🎯 تحلیل نتایج

### ✔️ **SVM، Random Forest، Logistic Regression**  
بهترین عملکرد را دارند و معمولاً به دقت ۱۰۰٪ می‌رسند.

### ✔️ **Decision Tree**  
گاهی ۱۰۰٪ می‌شود، ولی ممکن است Overfit کند.

### ✔️ **Naive Bayes**  
سریع و سبک است، ولی کمی دقت پایین‌تر دارد چون فرض استقلال ویژگی‌ها در Iris کاملاً برقرار نیست.

---

# 🌱 جمع‌بندی

- دیتاست Iris برای اکثر مدل‌ها **بسیار آسان** است.  
- بهترین مدل‌ها معمولاً **SVM** و **Random Forest** هستند.  
- اگر سرعت مهم باشد، **Naive Bayes** انتخاب خوبی است.  
- اگر بخواهی مدل‌ها را روی ۲ فیچر تست کنیم، نتایج کمی متفاوت می‌شود.

---
