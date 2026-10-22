import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Purchase Intelligence | Cart Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "discount_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.json"

# --------------------------------------------------
# PROFESSIONAL STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1120;
        --panel: #121c2d;
        --border: #27364b;
        --muted: #9aaac0;
        --accent: #60a5fa;
        --text: #edf4ff;
    }

    .stApp {
        background: #0b1120;
        color: var(--text);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #142344 0%, #1e4f83 100%);
        border: 1px solid #315a88;
        border-radius: 18px;
        padding: 30px 32px;
        margin-bottom: 26px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.16);
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(1.8rem, 3vw, 2.65rem);
        font-weight: 750;
        letter-spacing: -0.7px;
        line-height: 1.2;
        margin: 0;
    }

    .hero p {
        color: #d5e5fb;
        font-size: 1rem;
        line-height: 1.65;
        margin: 13px 0 0 0;
    }

    .eyebrow {
        color: #93c5fd;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    h2, h3 {
        color: #edf4ff !important;
        letter-spacing: -0.3px;
    }

    .section-intro {
        color: #a9b9ce;
        font-size: 0.94rem;
        margin-top: -5px;
        margin-bottom: 20px;
    }

    div[data-testid="stForm"] {
        background: #101a2a;
        border: 1px solid #27364b;
        border-radius: 16px;
        padding: 22px;
    }

    div[data-testid="stWidgetLabel"] p {
        color: #dce8f8;
        font-weight: 550;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 9px;
    }

    .stButton button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        min-height: 48px;
        font-weight: 650;
        transition: 0.2s ease;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(100deg, #2563eb, #3b82f6);
        color: #ffffff;
        border: none;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(100deg, #1d4ed8, #2563eb);
        border: none;
    }

    .result-panel {
        background: linear-gradient(145deg, #121e31, #101827);
        border: 1px solid #2b3b52;
        border-radius: 15px;
        padding: 22px;
        min-height: 135px;
    }

    .result-label {
        color: #a9bad0;
        font-size: 0.86rem;
        margin-bottom: 12px;
    }

    .result-value {
        color: #f3f7ff;
        font-size: clamp(1.65rem, 3vw, 2.2rem);
        font-weight: 750;
        letter-spacing: -0.6px;
    }

    .result-caption {
        color: #91a5be;
        font-size: 0.82rem;
        margin-top: 8px;
    }

    .recommendation {
        background: #13283a;
        border: 1px solid #28516e;
        border-left: 4px solid #60a5fa;
        border-radius: 12px;
        padding: 19px 21px;
        margin: 15px 0 22px 0;
    }

    .recommendation-title {
        color: #e6f2ff;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .recommendation-text {
        color: #c5d5e8;
        line-height: 1.7;
    }

    .footer {
        border-top: 1px solid #27364b;
        margin-top: 35px;
        padding-top: 18px;
        color: #8294ab;
        font-size: 0.82rem;
        line-height: 1.7;
    }

    [data-testid="stMetric"] {
        background: #121c2d;
        border: 1px solid #27364b;
        border-radius: 12px;
        padding: 16px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 0.8rem 2rem 0.8rem;
        }

        .hero {
            padding: 23px 20px;
        }

        div[data-testid="stForm"] {
            padding: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HEADER
# The repeated subtitle and blue information banner
# have intentionally been removed.
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning · E-commerce Analytics</div>
        <h1>🛒 Cart Abandonment &amp; Purchase Prediction</h1>
        <p>
            Understand shopping-session behaviour, estimate purchase
            likelihood, and support smarter customer engagement decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)
    return model, feature_columns


if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    st.error(
        "Required model files are missing. Make sure discount_model.pkl "
        "and feature_columns.json are in the same repository folder as app.py."
    )
    st.stop()

try:
    model, feature_columns = load_artifacts()
except Exception as exc:
    st.error(
        "The model could not be loaded. Check the model file and ensure "
        "scikit-learn matches the version used to train the model."
    )
    st.code(str(exc))
    st.stop()

# --------------------------------------------------
# SESSION INPUTS
# --------------------------------------------------
numeric_inputs = {
    "Administrative": ("Administrative pages visited", 0, 0, 20),
    "Administrative_Duration": ("Administrative time (seconds)", 0.0, 0.0, 5000.0),
    "Informational": ("Informational pages visited", 0, 0, 20),
    "Informational_Duration": ("Informational time (seconds)", 0.0, 0.0, 5000.0),
    "ProductRelated": ("Product-related pages visited", 5, 0, 500),
    "ProductRelated_Duration": ("Product-related time (seconds)", 100.0, 0.0, 50000.0),
    "BounceRates": ("Bounce rate", 0.02, 0.0, 1.0),
    "ExitRates": ("Exit rate", 0.05, 0.0, 1.0),
    "PageValues": ("Page value", 0.0, 0.0, 10000.0),
    "SpecialDay": ("Special-day proximity (0 to 1)", 0.0, 0.0, 1.0),
    "OperatingSystems": ("Operating system code", 2, 1, 8),
    "Browser": ("Browser code", 2, 1, 13),
    "Region": ("Region code", 1, 1, 9),
    "TrafficType": ("Traffic source code", 2, 1, 20),
    "Weekend": ("Weekend session", False, None, None),
}

categorical_inputs = {
    "Month": (
        "Month",
        ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ),
    "VisitorType": (
        "Visitor type",
        ["Returning_Visitor", "New_Visitor", "Other"],
    ),
}

st.header("1. Shopping session details")
st.markdown(
    '<div class="section-intro">Enter the available session details to estimate the likelihood of a completed purchase.</div>',
    unsafe_allow_html=True,
)

input_values = {}

with st.form("prediction_form"):
    left, right = st.columns(2)
    numeric_items = list(numeric_inputs.items())
    split_at = (len(numeric_items) + 1) // 2

    for col, items in zip(
        [left, right],
        [numeric_items[:split_at], numeric_items[split_at:]],
    ):
        with col:
            for feature, config in items:
                label, default, minimum, maximum = config

                if feature == "Weekend":
                    input_values[feature] = st.selectbox(
                        label,
                        options=[False, True],
                        format_func=lambda value: "Yes" if value else "No",
                        key=feature,
                    )
                elif isinstance(default, int):
                    input_values[feature] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1,
                        key=feature,
                    )
                else:
                    input_values[feature] = st.number_input(
                        label,
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=float(default),
                        step=0.01,
                        key=feature,
                    )

    cat_left, cat_right = st.columns(2)

    for col, feature in zip(
        [cat_left, cat_right],
        categorical_inputs,
    ):
        with col:
            label, options = categorical_inputs[feature]
            input_values[feature] = st.selectbox(
                label,
                options,
                key=feature,
            )

    submitted = st.form_submit_button(
        "🔍  Predict Purchase Probability",
        use_container_width=True,
    )

# --------------------------------------------------
# PREDICTION RESULTS
# --------------------------------------------------
if submitted:
    input_df = pd.DataFrame([input_values])

    missing = [column for column in feature_columns if column not in input_df.columns]
    extra = [column for column in input_df.columns if column not in feature_columns]

    if missing or extra:
        st.error(
            f"Input columns do not match the trained model. "
            f"Missing: {missing}; unexpected: {extra}"
        )
        st.stop()

    input_df = input_df[feature_columns]

    try:
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        purchase_index = classes.index(1)
        no_purchase_index = classes.index(0)

        purchase_probability = float(probabilities[purchase_index])
        no_purchase_probability = float(probabilities[no_purchase_index])

        predicted_class = int(model.predict(input_df)[0])
        is_purchase = predicted_class == 1

    except Exception as exc:
        st.error("Prediction failed. Please check the model and input features.")
        st.code(str(exc))
        st.stop()


    st.divider()
    st.header("2. Prediction results")
    st.markdown(
        '<div class="section-intro">Estimated probabilities for this shopping session.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="result-panel">
                <div class="result-label">Purchase probability</div>
                <div class="result-value">{purchase_probability:.1%}</div>
                <div class="result-caption">Estimated likelihood of purchase</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="result-panel">
                <div class="result-label">No-purchase probability</div>
                <div class="result-value">{no_purchase_probability:.1%}</div>
                <div class="result-caption">Estimated likelihood of no purchase</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        prediction_label = "Purchase" if is_purchase else "No purchase"
        st.markdown(
            f"""
            <div class="result-panel">
                <div class="result-label">Model prediction</div>
                <div class="result-value">{prediction_label}</div>
                <div class="result-caption">Predicted class from the trained model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(purchase_probability)

    st.subheader("Business recommendation")

    if is_purchase :
        recommendation_title = "Purchase likely"
        recommendation_text = (
            "The model predicts that this session is more likely to result "
            "in a purchase. Avoid offering an automatic discount based only "
            "on this prediction; use your normal marketing eligibility rules."
        )
    else:
        recommendation_title = "Potential non-conversion"
        recommendation_text = (
            "The model predicts that this session is more likely not to "
            "result in a purchase. Consider a reminder or a carefully measured "
            "discount experiment for eligible shoppers. This prediction alone "
            "does not prove that a discount will change the outcome."
        )

    st.markdown(
        f"""
        <div class="recommendation">
            <div class="recommendation-title">{recommendation_title}</div>
            <div class="recommendation-text">{recommendation_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Business interpretation")
    st.write(
        f"For the entered session, the model estimates a "
        f"{purchase_probability:.1%} probability of purchase and a "
        f"{no_purchase_probability:.1%} probability of no purchase. "
        "These are estimates, not guarantees."
    )

# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------
with st.expander("About this project"):
    st.markdown(
        """
        **Objective:** Predict whether an online shopping session results
        in a purchase using a Random Forest classifier.

        **Dataset:** UCI Online Shoppers Purchasing Intention.

        **Business application:** Help e-commerce teams identify sessions
        with lower predicted purchase likelihood and evaluate targeted
        customer-engagement strategies.

        **Important limitation:** The dataset records whether a purchase
        occurred during a session. It does not directly confirm that a
        shopper abandoned a cart, and predictions alone cannot establish
        that discounts cause more purchases.
        """
    )

st.markdown(
    """
    <div class="footer">
        <strong>Purchase Intelligence</strong> · Random Forest classification<br>
        Academic case study · Predictions are estimates and should be
        validated before real-world business use.
    </div>
    """,
    unsafe_allow_html=True,
)
