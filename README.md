# 📌 Week 2 – Machine Learning Assignment  
### **Drug Response Prediction Using Random Forest (Simulated Dataset)**  
**Prepared by:** Betül Şen  

---

## 📍 Overview  
This assignment demonstrates how machine learning can be used to **predict cancer cell drug response** (IC50 values) using **simulated gene expression data**.  
A full data science workflow is implemented:

- Creating synthetic datasets  
- Preprocessing and scaling  
- Training a Random Forest model  
- Hyperparameter tuning (GridSearchCV)  
- Model evaluation  
- Example predictions  

This serves as a simplified version of real biomedical ML pipelines such as **GDSC, CTRP, CCLE**, etc.

---

## 📊 Workflow Structure  
1. **Generate Fake Gene Expression Data** (100 cell lines × 500 genes)  
2. **Generate Fake Drug Response Values** for 5 different drugs  
3. **Select One Drug (Cisplatin in this example)**  
4. **Match expression and drug response data**  
5. **Split into train/test sets**  
6. **Scale the data**  
7. **Train a Random Forest Regressor**  
8. **Evaluate using MSE, R²**  
9. **Make example predictions**  

---

## 📂 Files  
| File | Description |
|------|-------------|
| `analysis.py` | Full machine learning implementation |
| `README.md` | Documentation and explanation |

---

## 🚀 How to Run  
1. Open a terminal inside the folder:
   ```bash
   python analysis.py
