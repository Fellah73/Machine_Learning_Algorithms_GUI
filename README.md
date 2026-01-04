# 🧠 Machine_Learning_Algorithms_GUI — Interactive Workflow in Python

Machine_Learning_Algorithms_GUI is a Python-based MVC application for exploring both **supervised** and **unsupervised machine learning algorithms**.  
It guides users step by step: dataset upload, preprocessing, learning type selection, algorithm choice, visualization, and comparison.

---

# 🔎 **Overview**  
This project provides a hands-on environment to experiment with **machine learning techniques**.  
The GUI is designed for clarity and pedagogy: each step of the workflow is implemented and visualized interactively.  
Built entirely with Python in a local environment, the app follows an **MVC architecture** for modularity and extensibility.

---

# 🔑 **Key highlights:**
- Upload `.csv` datasets directly into the GUI.
- Preprocess data: handle missing values, detect outliers, normalize features.
- Select the **learning type**:  
  - **Supervised**: KNN, Naive Bayes, C4.5  
  - **Unsupervised**: K-Means, K-Medoids, DIANA, AGNES, DBSCAN
- Choose the algorithm within the selected type.
- Visualize results with scatter plots (partitioning & density), dendrograms (hierarchical), or classification plots (supervised).
- Compare algorithms:  
  - Within the same type  
  - Across all types  
- Evaluate metrics: **Silhouette Score**, intra/inter-cluster distances, and supervised accuracy metrics.

---

# 🚀 **Features**

📂 **Step 1 — Dataset Upload**  
- Import `.csv` files via the GUI.  
- Preview dataset structure before processing.  

⚙️ **Step 2 — Preprocessing**  
- Analyze and replace missing values.  
- Detect and handle outliers.  
- Normalize features for consistent scaling.  

🔀 **Step 3 — Learning Type Selection**  
- Choose between **Supervised** or **Unsupervised** learning.  
- Supervised: KNN, Naive Bayes, C4.5.  
- Unsupervised: K-Means, K-Medoids, DIANA, AGNES, DBSCAN.  

🔬 **Step 4 — Algorithm Selection**  
- Select the specific algorithm within the chosen type.  

👁 **Step 5 — Visualization**  
- Scatter plots for partitioning & density algorithms.  
- Dendrograms for hierarchical algorithms.  
- Classification plots for supervised algorithms.  

🆚 **Step 6 — Comparison**  
- Compare algorithms of the same type.  
- Compare across supervised and unsupervised algorithms.  
- Metrics: Silhouette Score, intra/inter-cluster distances, supervised accuracy & precision.  

---

# 🛠️ **Technologies Used**
- **Python (3.x)** — core language  
- **Tkinter** — GUI framework  
- **scikit-learn** — supervised & unsupervised algorithms, metrics  
- **pandas** — dataset handling  
- **numpy** — numerical operations  
- **matplotlib / seaborn** — visualization  
- **MVC architecture** — structured application design  

---

# 💻 **Tech stack**
- Python (3.x) — main environment  
- scikit-learn — ML algorithms & evaluation  
- pandas + numpy — data manipulation  
- matplotlib + seaborn — plotting  
- Tkinter — GUI interface  
- MVC — application architecture  

---

🚀 **Getting Started**
1. Clone the repository.  
2. Install dependencies:  
   ```bash
   git clone your-repo
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
