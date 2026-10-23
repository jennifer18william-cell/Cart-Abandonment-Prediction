import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "discount_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

st.set_page_config(
    page_title="Discount Recommendation Classifier",
    page_icon="🛍️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

model = load_model()
metrics = load_metrics()

st.title("🛍️ Cart Session Discount Classifier")
st.write(
    "Enter a shopping-session profile to predict the dataset's "
    "`should_receive_discount` label."
)
st.warning(
    "Important: this project predicts the supplied dataset label, not verified cart abandonment "
    "or the causal effect of offering a discount. The original meaning of labels 0 and 1 has not "
    "been confirmed by the dataset documentation."
)

with st.sidebar:
    st.header("About this model")
    st.write("Algorithm: Random Forest classifier")
    st.write("Training data: 1,000 sessions supplied with the case study")
    if metrics:
        st.metric("Held-out test accuracy", f"{metrics.get('accuracy', 0):.1%}")
    st.caption("The reported test score is from one stratified 80/20 split (random_state=42).")

# Defaults are representative demo values, not recommended discount rules.
with st.form("session_form"):
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        annual_income = st.number_input("Annual income", min_value=0.0, value=50000.0, step=1000.0)
        gender = st.selectbox("Gender code", [0, 1], help="Uses the numeric codes present in the dataset.")
        device_mobile = st.selectbox("Using mobile device?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        traffic_source = st.selectbox("Traffic source code", [1, 2, 3, 4])
    with c2:
        product_category = st.selectbox("Product category code", [0, 1, 2, 3])
        current_session_duration_sec = st.number_input("Session duration (seconds)", min_value=0.0, value=180.0, step=10.0)
        time_spent_deviation = st.number_input("Time-spent deviation", value=0.0, step=1.0)
        pages_viewed = st.number_input("Pages viewed", min_value=0, max_value=1000, value=4)
        basket_icon_click = st.selectbox("Clicked basket icon?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with c3:
        cart_items_count_this_session = st.number_input("Cart items this session", min_value=0, max_value=100, value=1)
        cart_total_value = st.number_input("Cart total value", min_value=0.0, value=50.0, step=5.0)
        scrolled_to_reviews = st.selectbox("Scrolled to reviews?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        checked_delivery_pdp = st.selectbox("Checked delivery information?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        saw_checkout = st.selectbox("Saw checkout?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.subheader("Customer history and engagement")
    d1, d2, d3 = st.columns(3)
    with d1:
        sign_in = st.selectbox("Signed in?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_returning_user = st.selectbox("Returning user?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        lifetime_purchases_count = st.number_input("Lifetime purchases", min_value=0, max_value=10000, value=2)
    with d2:
        customer_conversion_rate = st.number_input("Historical conversion rate", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.3f")
        days_since_last_discount = st.number_input(
            "Days since last discount (999 if dataset placeholder applies)",
            min_value=0, max_value=5000, value=999
        )
        previous_review_sentiment = st.selectbox("Previous review sentiment code", [0, 1, 2, 3])
    with d3:
        social_channels_followed_count = st.number_input("Social channels followed", min_value=0, max_value=1000, value=2)

    submitted = st.form_submit_button("Predict label", type="primary", use_container_width=True)

if submitted:
    row = {
        "age": age,
        "gender": str(gender),
        "annual_income": annual_income,
        "device_mobile": str(device_mobile),
        "traffic_source": str(traffic_source),
        "product_category": str(product_category),
        "current_session_duration_sec": current_session_duration_sec,
        "time_spent_deviation": time_spent_deviation,
        "pages_viewed": pages_viewed,
        "basket_icon_click": str(basket_icon_click),
        "cart_items_count_this_session": cart_items_count_this_session,
        "cart_total_value": cart_total_value,
        "scrolled_to_reviews": str(scrolled_to_reviews),
        "checked_delivery_pdp": str(checked_delivery_pdp),
        "saw_checkout": str(saw_checkout),
        "sign_in": str(sign_in),
        "is_returning_user": str(is_returning_user),
        "lifetime_purchases_count": lifetime_purchases_count,
        "customer_conversion_rate": customer_conversion_rate,
        "days_since_last_discount": days_since_last_discount,
        "previous_review_sentiment": str(previous_review_sentiment),
        "social_channels_followed_count": social_channels_followed_count,
    }
    input_df = pd.DataFrame([row])
    label = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    class_labels = list(model.named_steps["classifier"].classes_)
    probability_by_class = {int(cls): float(prob) for cls, prob in zip(class_labels, probabilities)}

    st.subheader("Prediction")
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Predicted dataset label", str(label))
    with p2:
        st.metric(f"Model probability for label {label}", f"{probability_by_class[label]:.1%}")
    st.write("Probability by label")
    st.progress(probability_by_class.get(1, 0.0), text=f"Label 1: {probability_by_class.get(1, 0.0):.1%}")
    st.progress(probability_by_class.get(0, 0.0), text=f"Label 0: {probability_by_class.get(0, 0.0):.1%}")
    st.caption(
        "These probabilities are model estimates for the supplied label and may not be calibrated. "
        "Do not use them as proof that a discount will increase conversion."
    )
    st.download_button(
        "Download input record as CSV",
        data=input_df.to_csv(index=False).encode("utf-8"),
        file_name="predicted_customer_session.csv",
        mime="text/csv"
    )

with st.expander("Model evaluation details"):
    if metrics:
        st.write(f"Test records: {metrics.get('test_size')}")
        st.write(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
        st.write("Confusion matrix (rows = actual labels 0, 1; columns = predicted labels 0, 1):")
        st.dataframe(pd.DataFrame(
            metrics.get("confusion_matrix", []),
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        ))
        report = metrics.get("classification_report", {})
        rows = []
        for label in ["0", "1", "macro avg", "weighted avg"]:
            if label in report:
                rows.append({"Class": label, **report[label]})
        st.dataframe(pd.DataFrame(rows).set_index("Class"))
    st.info("A held-out test split measures prediction of the existing labels. It does not validate the label definition or demonstrate business impact.")
