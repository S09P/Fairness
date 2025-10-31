import streamlit as st
import pandas as pd
from your_functions import Worker, Worker_1  # Ensure both functions exist in your_functions.py

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Fair ML App", layout="wide")

# ---------------- BACKGROUND + TYPING HEADING ----------------
st.markdown("""
    <style>
        body {
            background: radial-gradient(circle at 10% 20%, #0a0f24, #081229 80%);
            color: white;
        }
        .typing {
            width: 36ch;
            animation: typing 5s steps(36) infinite alternate, blink 0.5s step-end infinite alternate;
            white-space: nowrap;
            overflow: hidden;
            border-right: 3px solid #00FFAA;
            font-size: 32px;
            font-weight: 700;
            color: #00FFAA;
            font-family: 'Courier New', monospace;
            display: inline-block;
        }
        @keyframes typing {
            0% { width: 0; }
            100% { width: 36ch; }
        }
        @keyframes blink {
            50% { border-color: transparent; }
        }
    </style>

    <div style='text-align:center; margin-top:20px; margin-bottom:20px;'>
        <div class='typing'>Generate Fair and Balanced Datasets with Ease</div>
    </div>
""", unsafe_allow_html=True)

# ---------------- MAIN TITLE ----------------
st.title("⚖️ Fair ML App")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("👀 Preview of Uploaded Dataset")
    st.dataframe(df.head())
    st.write(f"**Original Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # ---------------- USER INPUTS ----------------
    protected_attr = st.selectbox("🧬 Select Protected Attribute", df.columns)
    target_var = st.selectbox("🎯 Select Target Variable (Label Column)", df.columns)

    # ---------------- PRIORITY SELECTION ----------------
    st.markdown("### ⚙️ Choose Processing Priority")
    priority = st.radio(
        "Which is your main goal?",
        ("Maximize Fairness", "Generate More Synthetic Data"),
        help="Choose whether to focus on fairness improvement or more synthetic sample generation."
    )

    # ---------------- PROCESS BUTTON ----------------
    if st.button("🚀 Run Fair Pipeline"):
        with st.spinner("Processing... This may take a few moments ⏳"):
            try:
                # Rename Target Column to 'Probability' for consistency
                df = df.rename(columns={target_var: "Probability"})

                # Check numeric columns
                non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
                if non_numeric_cols:
                    st.error(
                        f"❌ Dataset contains non-numeric columns: {', '.join(non_numeric_cols)}.\n"
                        "Please ensure all features are numeric before running the Fair pipeline."
                    )
                    st.stop()

                # ---------------- RUN SELECTED WORKER ----------------
                if priority == "Maximize Fairness":
                    processed_df, metrics_df = Worker(df, protected_attr)
                else:
                    processed_df, metrics_df = Worker_1(df, protected_attr)

                # Rename back
                processed_df = processed_df.rename(columns={"Probability": target_var})

                # ---------------- SHOW RESULTS ----------------
                st.subheader("📊 Original vs Processed Metrics")
                st.dataframe(metrics_df)
                st.write(f"**Processed Dataset Shape:** {processed_df.shape[0]} rows × {processed_df.shape[1]} columns")
                

# Show synthetic data generation ratio
                generation_ratio = processed_df.shape[0] / df.shape[0]
                st.info(f"🧠 New Dataset Size :- **{generation_ratio:.2f}×** the original size ")


                st.subheader("🧩 Processed Dataset (Preview)")
                st.dataframe(processed_df.head())

                # ---------------- DOWNLOAD BUTTON ----------------
                csv = processed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Processed Dataset as CSV",
                    data=csv,
                    file_name="processed_dataset.csv",
                    mime="text/csv"
                )

                st.success("✅ Fair pipeline completed successfully!")

            except Exception as e:
                st.error(f"⚠️ Error while processing dataset: {e}")
else:
    st.info("👆 Upload a CSV file to get started!")
