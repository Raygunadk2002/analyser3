
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyexcel
import os

st.set_page_config(layout="wide")
st.title("📐 Structural Movement Graph Analyser v3")

st.info("🧠 VERSION: v3 — Mapping columns + preview safe rendering")

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
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.dropna(how="all", axis=1)

        st.success("File loaded.")
        try:
            st.dataframe(df.head())
        except Exception as e:
            st.warning("Could not preview table. Error likely from incompatible cell types.")
            st.text(f"Preview error: {e}")

        st.subheader("🧭 Map Columns")

        col_date = st.selectbox("Select the date/time column", df.columns)
        col_sensor = st.selectbox("Select the sensor output column", df.columns)
        col_temp = st.selectbox("Select the temperature column (optional)", ['None'] + list(df.columns))

        try:
            df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
            df = df.dropna(subset=[col_date])
            df = df.sort_values(col_date)
            df[col_sensor] = pd.to_numeric(df[col_sensor], errors='coerce')
            if col_temp != 'None':
                df[col_temp] = pd.to_numeric(df[col_temp], errors='coerce')

            fig, ax = plt.subplots()
            ax.plot(df[col_date], df[col_sensor], label='Sensor Value')
            if col_temp != 'None':
                ax2 = ax.twinx()
                ax2.plot(df[col_date], df[col_temp], color='orange', alpha=0.5, label='Temperature')
            ax.set_title("Sensor Data Over Time")
            st.pyplot(fig)

            classification_result = classify_pattern_ml(df[col_sensor], df[col_temp] if col_temp != 'None' else None)
            st.subheader("🤖 Pattern Classification")
            st.write("📊", ", ".join(classification_result))

            if st.button("Export PDF Report"):
                pdf_path = export_pdf_report(df, col_date, col_sensor, col_temp, classification_result)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(label="Download PDF", data=pdf_file, file_name="graph_report_v3.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"Processing error: {e}")
    else:
        st.error("No valid data found in the uploaded file.")
