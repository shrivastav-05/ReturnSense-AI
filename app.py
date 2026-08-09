from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and preprocessor
model = joblib.load("return_prediction_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get input values from HTML
    category = request.form["category"]
    price = float(request.form["price"])
    discount = float(request.form["discount"])
    quantity = int(request.form["quantity"])

    payment_method = request.form["payment_method"]
    region = request.form["region"]

    shipping_cost = float(request.form["shipping_cost"])
    profit_margin = float(request.form["profit_margin"])

    customer_age = int(request.form["customer_age"])
    customer_gender = request.form["customer_gender"]

    # Calculate Total Amount
    total_amount = price * quantity * (1 - discount / 100)

    # Create DataFrame
    new_data = pd.DataFrame({
        "category": [category],
        "price": [price],
        "discount": [discount],
        "quantity": [quantity],
        "payment_method": [payment_method],
        "region": [region],
        "total_amount": [total_amount],
        "shipping_cost": [shipping_cost],
        "profit_margin": [profit_margin],
        "customer_age": [customer_age],
        "customer_gender": [customer_gender]
    })

    # Preprocess input
    transformed_data = preprocessor.transform(new_data)

    # Prediction
    prediction = model.predict(transformed_data)

    # Prediction probability
    probability = model.predict_proba(transformed_data)

    not_returned_probability = probability[0][0] * 100
    returned_probability = probability[0][1] * 100

    # Result
    if prediction[0] == 1:

        result = "Product is likely to be Returned"
        status = "danger"

    else:

        result = "Product is NOT likely to be Returned"
        status = "success"

    return render_template(
        "index.html",
        result=result,
        status=status,
        probability=returned_probability,
        not_returned_probability=not_returned_probability,
        returned_probability=returned_probability,
        total_amount=total_amount
    )


if __name__ == "__main__":
    app.run(debug=True)