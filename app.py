                        values[feature] = st.number_input(
                            label, min_value=minimum, max_value=maximum,
                            value=default, step=step, key=f"input_{feature}"
                        )
                    else:
                        values[feature] = st.number_input(
                            label, min_value=float(minimum), max_value=float(maximum),
                            value=float(default), step=float(step), key=f"input_{feature}"
                        )

    with tab_context:
        c1, c2 = st.columns(2)
        with c1:
            values["Month"] = st.selectbox("Shopping month", categorical_inputs["Month"])
            values["VisitorType"] = st.selectbox("Visitor type", categorical_inputs["VisitorType"])
        with c2:
            values["Weekend"] = st.radio("Was this a weekend session?", [False, True],
                                          format_func=lambda x: "Yes" if x else "No",
                                          horizontal=True)

    submitted = st.form_submit_button("✨ Predict session outcome", use_container_width=True)

# --------------------------- PREDICTION -------------------
if submitted:
    input_df = pd.DataFrame([values])
    missing = [c for c in feature_columns if c not in input_df.columns]
    extra = [c for c in input_df.columns if c not in feature_columns]

    if missing or extra:
        st.error(
            "The input fields do not match the model's training features. "
            f"Missing: {missing or 'None'} · Unexpected: {extra or 'None'}"
        )
        st.stop()

    input_df = input_df[feature_columns]

    try:
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        purchase_probability = float(probabilities[classes.index(1)])
        no_purchase_probability = float(probabilities[classes.index(0)])
        predicted_class = int(model.predict(input_df)[0])
    except Exception as exc:
        st.error("Could not generate a prediction. Check model compatibility and feature types.")
        st.exception(exc)
        st.stop()

    st.divider()
    st.header("2 · Prediction summary")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Purchase likelihood</div>'
            f'<div class="metric-value">{purchase_probability:.1%}</div>'
            f'<div class="metric-note">Estimated chance of purchase</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">No-purchase likelihood</div>'
            f'<div class="metric-value">{no_purchase_probability:.1%}</div>'
            f'<div class="metric-note">Estimated chance of no purchase</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        label = "Purchase" if predicted_class == 1 else "No purchase"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Predicted outcome</div>'
            f'<div class="metric-value">{label}</div>'
            f'<div class="metric-note">Model classification</div></div>',
            unsafe_allow_html=True,
        )

    st.write("Purchase probability scale")
    st.progress(purchase_probability)

    st.subheader("3 · Suggested business action")
    if predicted_class == 1:
        st.markdown(
            """
            <div class="advice">
              <strong>Purchase appears more likely</strong><br>
              Consider using a standard follow-up journey rather than automatically
              offering a discount. Evaluate campaign eligibility and profitability first.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="advice">
              <strong>Potential non-conversion</strong><br>
              Consider a reminder message or test a targeted incentive with eligible
              shoppers. Measure the results with an experiment before rolling it out.
            </div>
            """,
            unsafe_allow_html=True,
        )

    result_df = pd.DataFrame([{
        "Purchase probability": purchase_probability,
        "No-purchase probability": no_purchase_probability,
        "Predicted outcome": label,
    }])
    st.download_button(
        "Download prediction summary (CSV)",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="cartsense_prediction.csv",
        mime="text/csv",
    )

# --------------------------- PROJECT NOTES -----------------
with st.expander("Project notes and limitations"):
    st.markdown(
        """
        - **Goal:** Estimate purchase completion from online shopping-session features.
        - **Model:** The application loads the previously trained model from `discount_model.pkl`.
        - **Feature mapping:** `feature_columns.json` determines the exact input order.
        - **Interpretation:** The model predicts an outcome; it does not prove a shopper abandoned a cart.
        - **Discount decisions:** A prediction alone cannot establish that offering a discount will cause a purchase.
        """
    )

st.markdown(
    '<div class="footer">CartSense · Random Forest classification · Academic demonstration</div>',
