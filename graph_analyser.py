
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyexcel
import os

st.set_page_config(layout="wide")
st.title("📐 Structural Movement Graph Analyser v2")

# Version indicator
st.info("🧠 VERSION: v2 — Improved XLS handling + ML scaffolding")

uploaded_file = st.file_uploader("Upload an Excel file (.xls or .xlsx)", type=["xls", "xlsx"])

def safe_read_excel(file):
    try:
        if file.name.endswith('.xlsx'):
            return pd.read_excel(file, engine='openpyxl')
        elif file.name.endswith('.xls'):
            content = file.read()
            tmp_path = "/tmp/upload.xls"
            with open(tmp_path, "wb") as f:
                f.write(content)
            sheet = pyexcel.get_sheet(file_name=tmp_path)
            return pd.DataFrame(sheet.to_array()[1:], columns=sheet.row[0])
        else:
            return None
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None

def classify_pattern_ml(values, temperature=None):
    results = []
    values = pd.Series(values).dropna()
    if len(values) < 5:
        return ["insufficient data"]

    if abs(values.iloc[-1] - values.iloc[0]) > values.std() * 2:
        results.append("progressive")
    if temperature is not None:
        temp_corr = pd.Series(values).corr(pd.Series(temperature))
        if abs(temp_corr) > 0.6:
            results.append("thermal")
    fft_vals = np.fft.fft(values - values.mean())
    if np.max(np.abs(fft_vals[1:20])) > values.std() * 5:
        results.append("seasonal")
    if not results:
        results.append("none")
    return results

def export_pdf_report(dataframe, time_col, value_col, temp_col, classification):
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    import tempfile

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Structural Graph Report", ln=True)
    pdf.cell(200, 10, txt=f"Detected Pattern(s): {', '.join(classification)}", ln=True)

    fig, ax = plt.subplots()
    ax.plot(dataframe[time_col], dataframe[value_col], label='Sensor')
    if temp_col != 'None':
        ax2 = ax.twinx()
        ax2.plot(dataframe[time_col], dataframe[temp_col], color='orange', alpha=0.5, label='Temperature')
    ax.set_title("Sensor Graph")
    img_path = tempfile.mktemp(suffix=".png")
    plt.savefig(img_path)
    pdf.image(img_path, x=10, y=40, w=180)

    pdf_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(pdf_path)
    return pdf_path

if uploaded_file:
    df = safe_read_excel(uploaded_file)
    if df is not None and not df.empty:
        st.success("File loaded.")
        st.dataframe(df.head())

        time_col = st.selectbox("Time column", df.columns)
        value_col = st.selectbox("Sensor value column", df.columns)
        temp_col = st.selectbox("Temperature column (optional)", ['None'] + list(df.columns))

        try:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col])
            df = df.sort_values(time_col)
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            if temp_col != 'None':
                df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')

            fig, ax = plt.subplots()
            ax.plot(df[time_col], df[value_col], label='Sensor')
            if temp_col != 'None':
                ax2 = ax.twinx()
                ax2.plot(df[time_col], df[temp_col], color='orange', alpha=0.5)
            ax.set_title("Sensor Value Over Time")
            st.pyplot(fig)

            classification_result = classify_pattern_ml(df[value_col], df[temp_col] if temp_col != 'None' else None)
            st.subheader("ML-Based Pattern Classification")
            st.write("📊", ", ".join(classification_result))

            if st.button("Export PDF Report"):
                pdf_path = export_pdf_report(df, time_col, value_col, temp_col, classification_result)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(label="Download Report", data=pdf_file, file_name="graph_report_v2.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"Processing error: {e}")
    else:
        st.error("No valid data found in the uploaded file.")
