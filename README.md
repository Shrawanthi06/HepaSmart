#  HepaSmart: Hepatitis Survival Prediction App

HepaSmart is a **machine learning–powered web application** that predicts the **survival outcome of hepatitis patients** based on clinical and laboratory data.
It allows **individual patient prediction** through an input form and **batch prediction** by uploading a CSV file — all through an elegant **Streamlit interface**.

---

##  Features

* Predicts **"Live"** or **"Die"** outcomes for hepatitis patients using a trained ML model.
* Accepts **manual inputs** through an interactive form.
* Supports **CSV file upload** for **batch predictions**.
* Displays **visual charts** for insights and analysis.
* Built with **Streamlit**, **scikit-learn**, and **Matplotlib**.

---

## 📂 Project Structure

```
HepaSmart/
│
├── app.py                # Streamlit web app
├── hepatitis_b_model.pkl # Trained ML model
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

---

##  Installation & Setup

### 1️ Clone or Download this Repository

```bash
git clone https://github.com/yourusername/HepaSmart.git
cd HepaSmart
```

### 2️ Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows (PowerShell):**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

>  If PowerShell blocks the script, run:
>
> ```bash
> Set-ExecutionPolicy Unrestricted -Scope Process
> ```

### 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️ Run the Application

```bash
streamlit run app.py
```

Then open the provided local URL (e.g. `http://localhost:8501/`) in your browser.

---

##  Model Information

* **Model Used:** Random Forest Classifier
* **Training Dataset:** UCI Hepatitis Dataset (ID: 46)
* **Features:**

  ```
  Age, Sex, Steroid, Antivirals, Fatigue, Malaise, Anorexia, Liver Big,
  Liver Firm, Spleen Palpable, Spiders, Ascites, Varices,
  Bilirubin, Alk Phosphate, Sgot, Albumin, Histology
  ```
* **Target:** Survival (`Live` or `Die`)

---

##  How to Use

###  Individual Prediction

1. Enter the patient's medical details using the form.
2. Click **"Predict"**.
3. View the predicted outcome and feature visualization.

###  Batch Prediction

1. Upload a `.csv` file with the same 18 feature columns.
2. Click **"Run Batch Prediction"**.
3. See results for all records along with visual summaries.

---

##  Example CSV Format

| Age | Sex | Steroid | Antivirals | Fatigue | Malaise | Anorexia | Liver Big | Liver Firm | Spleen Palpable | Spiders | Ascites | Varices | Bilirubin | Alk Phosphate | Sgot | Albumin | Histology |
| --- | --- | ------- | ---------- | ------- | ------- | -------- | --------- | ---------- | --------------- | ------- | ------- | ------- | --------- | ------------- | ---- | ------- | --------- |
| 30  | 1   | 1       | 1          | 0       | 1       | 1        | 0         | 1          | 0               | 0       | 0       | 0       | 1.2       | 80            | 30   | 4.0     | 1         |

---

##  Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Libraries:** scikit-learn, pandas, numpy, matplotlib
* **Deployment Ready:** Can be hosted on Streamlit Cloud, Heroku, or AWS EC2

---

## 👩‍🔬 Author

**Shrawanthi P**
🔗 [LinkedIn](https://www.linkedin.com/in/shrawanthi-p-861533264)

---

Would you like me to add a **screenshot of the HepaSmart web app interface** (showing the form and CSV upload UI) to the README as well?
