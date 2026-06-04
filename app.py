from flask import Flask, request, render_template,session,redirect, url_for

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application
app.secret_key = "Preet@123"

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_data():
    if request.method == "GET":
        results = session.pop("results", None)
        return render_template("home.html", results=results)
    else:
        data = CustomData(
            no_of_dependents=float(request.form.get("no_of_dependents")),
            education=request.form.get("education"),
            self_employed=request.form.get("self_employed"),
            income_annum = float(request.form.get("income_annum")),
            loan_amount=float(request.form.get("loan_amount")),
            loan_term=float(request.form.get("loan_term")),
            cibil_score=float(request.form.get("cibil_score")),
            residential_assets_value=float(request.form.get("residential_assets_value")),
            commercial_assets_value=float(request.form.get("commercial_assets_value")),
            luxury_assets_value=float(request.form.get("luxury_assets_value")),
            bank_asset_value = float(request.form.get("bank_asset_value"))
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        session["results"] = results[0].strip()  
        return redirect(url_for("predict_data"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
