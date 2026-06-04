# Loan Approval Prediction

A Flask web application that predicts whether a loan application will be approved or rejected using a machine learning classification model.

---

## Project Structure

```
project/
├── app.py
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── templates/
│   ├── index.html       # Landing page
│   └── home.html        # Prediction form
└── static/
    └── css/
        └── style.css
```

---

## Setup

**1. Clone the repository**
```bash
git clone <repo-url>
cd project
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Requirements

```
flask
numpy
pandas
scikit-learn
scipy
```

---

## Input Features

| Feature | Description | Type |
|---|---|---|
| no_of_dependents | Number of dependents | 0 – 5 |
| education | Graduate / Not Graduate | Categorical |
| self_employed | Yes / No | Categorical |
| income_annum | Annual income | Numeric |
| loan_amount | Requested loan amount | Numeric |
| loan_term | Loan duration in months | Numeric |
| cibil_score | Credit score | 300 – 900 |
| residential_assets_value | Value of residential assets | Numeric |
| commercial_assets_value | Value of commercial assets | Numeric |
| luxury_assets_value | Value of luxury assets | Numeric |
| bank_asset_value | Value of bank assets | Numeric |

---

## Output

- **Approved** — loan is likely to be sanctioned
- **Rejected** — loan application may be declined

---

## Notes

- The app uses the PRG (Post-Redirect-Get) pattern to prevent form resubmission on page refresh.
- Predictions are stored temporarily in Flask session and cleared after display.
- This is an ML-based prediction and should not be treated as a final financial decision.