from preprocessing import BankMarketingPreprocessor
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import sys
import tempfile

# Add the parent directory to path to access files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocessing module

# Load the model and initialize preprocessor
model_path = '../best_banking_model.pkl'
model = joblib.load(model_path)
preprocessor = BankMarketingPreprocessor()


def predict_individual(age, job, marital, education, default, housing, loan, contact, month,
                       day_of_week, campaign, pdays, previous, poutcome, emp_var_rate,
                       cons_price_idx, cons_conf_idx, euribor3m, nr_employed):
    """
    Function to make prediction for an individual data point
    """
    # Create a dictionary for the input values
    data_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }

    # Process the raw inputs to match the encoded format
    processed_input = preprocessor.preprocess_individual(data_dict)

    # Make prediction
    prob = model.predict_proba(processed_input)[0][1]
    subscription_pred = "Yes" if prob > 0.5 else "No"

    return {
        'Subscription Prediction': subscription_pred,
        'Subscription Probability': f"{prob:.2%}"
    }


def predict_file(file):
    """
    Function to make predictions for a batch of data
    """
    try:
        df = pd.read_csv(file.name)

        # Check if the DataFrame has expected structure
        if len(df.columns) < 10:
            return None, f"Error: The uploaded file does not have enough columns. Expected format similar to the bank-additional-full.csv file."

        # Preprocess the dataframe
        processed_df = preprocessor.preprocess_batch(df)

        # Make predictions
        predictions_proba = model.predict_proba(processed_df)[:, 1]
        predictions = (predictions_proba > 0.5).astype(int)

        # Add predictions to the original dataframe
        df['subscription_prob'] = predictions_proba
        df['subscription_pred'] = ["Yes" if pred == 1 else "No" for pred in predictions]

        # Save the results to a temporary CSV file
        temp_dir = tempfile.gettempdir()
        result_path = os.path.join(temp_dir, "prediction_results.csv")
        df.to_csv(result_path, index=False)

        # Return the CSV file path and a summary message
        yes_count = sum(predictions)
        total = len(predictions)
        message = f"Predictions complete! {yes_count} out of {total} customers ({yes_count/total:.2%}) are predicted to subscribe."

        return result_path, message

    except Exception as e:
        return None, f"Error processing file: {str(e)}"


# Job categories from the original dataset
job_categories = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"
]

# Create the Gradio interface
with gr.Blocks(title="Banking Marketing Prediction") as demo:
    gr.Markdown("# Banking Marketing Subscription Predictor")
    gr.Markdown(
        "Predict if a customer will subscribe to a term deposit based on marketing campaign data.")

    with gr.Tabs():
        with gr.TabItem("Individual Prediction"):
            with gr.Row():
                with gr.Column():
                    age = gr.Slider(18, 95, value=40, label="Age")
                    job = gr.Dropdown(job_categories, label="Job Type")
                    marital = gr.Radio(
                        ["married", "single", "divorced"], label="Marital Status")
                    education = gr.Dropdown(
                        ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                         "professional.course", "university.degree", "unknown"],
                        label="Education Level"
                    )
                    default = gr.Radio(
                        ["yes", "no"], value="no", label="Has Credit in Default?")
                    housing = gr.Radio(
                        ["yes", "no"], value="no", label="Has Housing Loan?")
                    loan = gr.Radio(["yes", "no"], value="no",
                                    label="Has Personal Loan?")

                with gr.Column():
                    contact = gr.Radio(
                        ["cellular", "telephone"], label="Contact Communication Type")
                    month = gr.Dropdown(
                        ["jan", "feb", "mar", "apr", "may", "jun",
                            "jul", "aug", "sep", "oct", "nov", "dec"],
                        label="Last Contact Month"
                    )
                    day_of_week = gr.Dropdown(
                        ["mon", "tue", "wed", "thu", "fri"], label="Last Contact Day of Week")
                    campaign = gr.Slider(
                        1, 50, value=2, step=1, label="Number of Contacts During Campaign")
                    pdays = gr.Slider(
                        0, 999, value=999, step=1, label="Days Since Last Contact (999 means never contacted)")
                    previous = gr.Slider(
                        0, 20, value=0, step=1, label="Number of Contacts Before This Campaign")
                    poutcome = gr.Radio(
                        ["failure", "nonexistent", "success"], label="Outcome of Previous Campaign")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Economic Context Attributes")
                    emp_var_rate = gr.Slider(-5.0, 5.0, value=0.0,
                                             step=0.1, label="Employment Variation Rate")
                    cons_price_idx = gr.Slider(
                        90.0, 95.0, value=93.5, step=0.1, label="Consumer Price Index")
                    cons_conf_idx = gr.Slider(-55.0, -25.0, value=-40.0,
                                              step=0.1, label="Consumer Confidence Index")
                    euribor3m = gr.Slider(
                        0.5, 5.5, value=3.0, step=0.1, label="Euribor 3 Month Rate")
                    nr_employed = gr.Slider(
                        4900, 5300, value=5100, step=10, label="Number of Employees (thousands)")

            predict_btn = gr.Button("Predict Subscription")
            output = gr.JSON(label="Prediction Results")

            predict_btn.click(
                predict_individual,
                inputs=[age, job, marital, education, default, housing, loan,
                        contact, month, day_of_week, campaign, pdays, previous,
                        poutcome, emp_var_rate, cons_price_idx, cons_conf_idx,
                        euribor3m, nr_employed],
                outputs=output
            )

        with gr.TabItem("Batch Prediction"):
            gr.Markdown("### Upload a CSV file for batch predictions")
            gr.Markdown(
                "The file should have the same structure as the original banking dataset.")

            file_input = gr.File(label="Upload CSV file")
            batch_predict_btn = gr.Button("Run Batch Prediction")

            # Add file output component for downloadable results
            csv_output = gr.File(label="Download Prediction Results")
            batch_output = gr.Textbox(label="Prediction Results")

            batch_predict_btn.click(
                predict_file,
                inputs=file_input,
                outputs=[csv_output, batch_output]
            )

    gr.Markdown("### About This Model")
    gr.Markdown("""
    This modelis trained on the Bank Marketing dataset to predict whether a customer will subscribe to a term deposit.
        
    For batch predictions, upload a CSV file with the same structure as the original dataset.
    You can download the prediction results as a CSV file.
    """)

if __name__ == "__main__":
    demo.launch()
