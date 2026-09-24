import gradio as gr
import joblib
import numpy as np
import json

# Load saved model, scalers, and threshold
model = joblib.load('fraud_model.pkl')
scaler_amount = joblib.load('scaler_amount.pkl')
scaler_time = joblib.load('scaler_time.pkl')

with open('threshold.json', 'r') as f:
    threshold = json.load(f)['threshold']

# Real example transactions pulled from the test set
FRAUD_EXAMPLE = {
    "amount": 76.94, "time": 74159.0,
    "v": [-1.54878809850026, 1.80869795041448, -0.953509033832342, 2.21308539346999,
          -2.01572779170327, -0.913456844516923, -2.35601298316433, 1.19716896702387,
          -1.67837405659509, -3.53865023182429, 3.1020899271543, -3.99337305447702,
          -1.93741062327519, -3.82289410599595, 0.830970110708369, -2.47535885382925,
          -5.21187516766885, -0.413871678166879, 0.933262164554872, 0.390785963777347,
          0.855138263312025, 0.77474482148342, 0.0590371520063436, 0.343199807900813,
          -0.468937928609185, -0.278337986906642, 0.625922215184372, 0.395573378256676]
}

LEGIT_EXAMPLE = {
    "amount": 11.5, "time": 61290.0,
    "v": [1.2288211502379, -0.0634077165201056, 0.274145142235826, 0.647465021810117,
          -0.0481345611508765, 0.372073028593297, -0.22423058741343, 0.0799390492455152,
          0.640758817066441, -0.273053702248503, -1.25272793883718, 0.465078770741453,
          0.400502115321077, -0.292841860600363, -0.10177401599731, -0.399835897844616,
          0.0343356567914817, -0.783550254934187, 0.141344900433949, -0.0965659023514416,
          -0.129554448055005, -0.0837793282428063, -0.151661473916324, -0.700371597289218,
          0.598550164523483, 0.491409070563651, 0.0029892597250263, 0.0017822861144491]
}


def predict_fraud(amount, time, *v_features):
    scaled_amount = scaler_amount.transform([[amount]])[0][0]
    scaled_time = scaler_time.transform([[time]])[0][0]

    features = np.array(list(v_features) + [scaled_amount, scaled_time]).reshape(1, -1)

    prob = model.predict_proba(features)[0][1]
    is_fraud = prob >= threshold

    if is_fraud:
        return f"""
        <div style="padding:20px; border-radius:12px; background:#fdecea; border:1px solid #f5c6cb; text-align:center;">
            <h2 style="color:#c0392b; margin:0;">FRAUD DETECTED</h2>
            <p style="font-size:16px; margin-top:10px;">Fraud Probability: <b>{prob:.2%}</b></p>
            <p style="color:#888; font-size:13px;">Threshold used: {threshold:.2%}</p>
        </div>
        """
    else:
        return f"""
        <div style="padding:20px; border-radius:12px; background:#eafaf1; border:1px solid #a3e4c1; text-align:center;">
            <h2 style="color:#1e8449; margin:0;">Legitimate Transaction</h2>
            <p style="font-size:16px; margin-top:10px;">Fraud Probability: <b>{prob:.2%}</b></p>
            <p style="color:#888; font-size:13px;">Threshold used: {threshold:.2%}</p>
        </div>
        """


def load_fraud_example():
    return [FRAUD_EXAMPLE["amount"], FRAUD_EXAMPLE["time"]] + FRAUD_EXAMPLE["v"]


def load_legit_example():
    return [LEGIT_EXAMPLE["amount"], LEGIT_EXAMPLE["time"]] + LEGIT_EXAMPLE["v"]


with gr.Blocks(title="Credit Card Fraud Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Credit Card Fraud Detector
        Enter transaction details manually, or click a button below to load a real
        example transaction from the test dataset and see the model classify it.
        """
    )

    with gr.Row():
        load_fraud_btn = gr.Button("Load Real Fraud Example")
        load_legit_btn = gr.Button("Load Real Legit Example")

    with gr.Row():
        with gr.Column(scale=1):
            amount = gr.Number(label="Transaction Amount ($)", value=100.0)
            time = gr.Number(label="Time (seconds since first transaction)", value=50000.0)

            with gr.Accordion("Advanced: V1-V28 features (click to expand)", open=False):
                v_sliders = []
                with gr.Row():
                    for col in range(4):
                        with gr.Column():
                            for i in range(col * 7 + 1, col * 7 + 8):
                                if i <= 28:
                                    v_sliders.append(gr.Slider(-10, 10, value=0, label=f"V{i}"))

            submit_btn = gr.Button("Check Transaction", variant="primary")

        with gr.Column(scale=1):
            output = gr.HTML(label="Result")

    submit_btn.click(
        fn=predict_fraud,
        inputs=[amount, time] + v_sliders,
        outputs=output,
    )

    load_fraud_btn.click(
        fn=load_fraud_example,
        inputs=[],
        outputs=[amount, time] + v_sliders,
    )

    load_legit_btn.click(
        fn=load_legit_example,
        inputs=[],
        outputs=[amount, time] + v_sliders,
    )

import os
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))git add app.py
