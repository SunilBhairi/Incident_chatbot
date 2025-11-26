#!/usr/bin/env python3
# Streamlit Text → Pandas chatbot for CSV search + LangChain summarization, charts, PDF export

import streamlit as st
import pandas as pd
import os
import io
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI

# LangChain / LLM imports
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------------------------------------------------
# Load API KEY from .env
# ---------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY missing in .env file!")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# LangChain LLM (uses OPENAI_API_KEY from env)
lc_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# ---------------------------------------------------------
# Hardcoded CSV path
# ---------------------------------------------------------
CSV_PATH = "Incidents_p1_p2_missing_value_handeled_enriched.csv"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"❌ CSV file not found at path: {CSV_PATH}")
    st.stop()


st.set_page_config(page_title="Incident ChatBot", layout="wide")

st.title("🔍 Incident Search Chatbot ")
st.caption(f"Dataset loaded from: `{CSV_PATH}`")

st.write("### Data preview")
st.dataframe(df.head(), use_container_width=True)

# ---------------------------------------------------------
# Utility: basic safety checker for generated code
# ---------------------------------------------------------
BANNED_PATTERNS = [
    "import ",
    "open(",
    "os.",
    "subprocess",
    "sys.",
    "eval(",
    "exec(",
    "pickle",
    "builtins",
    "__class__",
    "__dict__",
    "__import__",
]

def is_code_safe(code: str):
    """Very simple static check to block obvious dangerous patterns."""
    lower_code = code.lower()
    for pat in BANNED_PATTERNS:
        if pat in lower_code:
            return False, pat
    return True, None

# ---------------------------------------------------------
# LangChain: Summary prompt + chain (for result summarization)
# ---------------------------------------------------------
summary_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert incident analyst. Use ONLY the context below to answer the user's question.
If the context doesn't contain the answer, make a reasonable guess or indicate missing info.

Context:
{context}

User Question:
{question}

Provide:
1. Short summary (2–4 lines)
2. Key facts (bullet list)
3. Suggested next steps (brief)
"""
)

summary_chain = LLMChain(
    llm=lc_llm,
    prompt=summary_prompt
)

MAX_CHARS_PER_CHUNK = 8000  # token-safe-ish chunking

def summarize_result_df(result_df: pd.DataFrame, user_query: str) -> str:
    """
    Token-safe summarization:
    - If no rows → return 'no context' message
    - If small → single LLM call
    - If large → chunk + summarize chunks + final summary over chunk summaries
    """
    if result_df is None or result_df.empty:
        return "No rows matched your query, so there is no incident context to summarize."

    context_text = result_df.to_string(index=False)

    # Small enough → one shot
    if len(context_text) <= MAX_CHARS_PER_CHUNK:
        return summary_chain.run({"context": context_text, "question": user_query})

    # Otherwise: chunk, summarize each, then summarize the summaries
    chunk_summaries = []
    for i in range(0, len(context_text), MAX_CHARS_PER_CHUNK):
        chunk = context_text[i : i + MAX_CHARS_PER_CHUNK]
        chunk_summary = summary_chain.run({"context": chunk, "question": user_query})
        chunk_summaries.append(f"Chunk {i // MAX_CHARS_PER_CHUNK + 1} summary:\n{chunk_summary}")

    combined_context = "\n\n".join(chunk_summaries)
    final_summary = summary_chain.run({"context": combined_context, "question": user_query})
    return final_summary

# ---------------------------------------------------------
# ⭐ NEW: Stage 3 – Root Cause Analysis (RCA)
# ---------------------------------------------------------
rca_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
You are a senior root-cause analyst.

Analyze the incident dataset provided below and generate:
1. Top 3 dominant root-cause patterns
2. Repeated failure themes across incidents
3. Clear probable root cause
4. 3–5 actionable recommendations to prevent future incidents

Context:
{context}

Provide output in structured markdown with:
### Major Patterns
### Probable RCA
### Recommendations
"""
)

rca_chain = LLMChain(
    llm=lc_llm,
    prompt=rca_prompt
)

def generate_rca_analysis(result_df: pd.DataFrame) -> str:
    """
    Generate a high-level RCA view from the filtered incidents.
    """
    if result_df is None or result_df.empty:
        return "No incidents available for RCA analysis."

    context_text = result_df.to_string(index=False)

    # Keep RCA context within a safe size
    if len(context_text) > MAX_CHARS_PER_CHUNK:
        context_text = context_text[:MAX_CHARS_PER_CHUNK]

    return rca_chain.run({"context": context_text})

# ---------------------------------------------------------
# PDF generation from summary + DataFrame
# ---------------------------------------------------------
def create_pdf(summary_text: str, result_df: pd.DataFrame) -> io.BytesIO:
    """
    Create a simple PDF with summary + top N rows of the result_df.
    Returns a BytesIO buffer ready for download.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    textobject = c.beginText(40, height - 40)
    textobject.setFont("Helvetica-Bold", 14)
    textobject.textLine("Incident Analysis Summary")
    textobject.moveCursor(0, 20)

    textobject.setFont("Helvetica", 10)
    textobject.textLine("")
    textobject.textLine("Summary:")
    textobject.textLine("")

    for line in summary_text.splitlines():
        textobject.textLine(line[:110])  # basic truncation per line

    textobject.textLine("")
    textobject.textLine("Top results (first 30 rows):")
    textobject.textLine("")

    limited_df = result_df.head(30)
    df_text = limited_df.to_string(index=False)
    for line in df_text.splitlines():
        if textobject.getY() < 40:  # new page if needed
            c.drawText(textobject)
            c.showPage()
            textobject = c.beginText(40, height - 40)
            textobject.setFont("Helvetica", 10)
        textobject.textLine(line[:110])

    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------------------------------------------------------
# LLM: Generate Pandas code from natural language
# ---------------------------------------------------------
def generate_pandas_code(user_query: str, column_list: str) -> str:
    system_prompt = f"""
You are an expert at converting English questions into Pandas filtering code.
You ALWAYS generate Python code that filters a DataFrame named df.

IMPORTANT RULES:
---------------------------------------
1. NEVER generate SQL.
2. NEVER generate markdown or explanatory text. ONLY Python code.
3. ALWAYS store final output in a DataFrame named: result
4. Wrap conditions inside parentheses.
5. Use boolean masks (mask_xxx) and combine them using '&' or '|'.
6. Do NOT read or write any files.
7. Do NOT import any libraries. Pandas is already imported as pd and df is in memory.
8. Do NOT call eval(), exec(), open(), or any OS/system functions.
9. Only operate on the existing DataFrame df.

---------------------------------------
### HUMAN ERROR LOGIC (MANDATORY WHEN APPLICABLE)
Use this exact pattern whenever the user query contains anything related to:
- human error
- operator error
- misconfiguration
- configuration issue
- runbook
- manual error
- human-error
- human issue

HUMAN ERROR CODE TEMPLATE (USE THIS FORMAT):

human_error_keywords = ['Human Error', 'human error', 'Operator Error',
                        'Misconfiguration', 'Configuration Issue',
                        'Runbook', 'Procedure', 'Manual error']

text_cols = ['Final Causation', 'Causation Code', 'Initial Causation',
             'Sub Category', 'Comments']

human_error_mask = pd.Series(False, index=df.index)

for col in text_cols:
    if col in df.columns:
        col_series = df[col].astype(str)
        for kw in human_error_keywords:
            human_error_mask = human_error_mask | col_series.str.contains(kw, case=False, na=False)

Example for "Human error incidents that caused P1 in LMS":

mask_p1 = (df['Priority'] == 'P1')
mask_lms = df['Product'].astype(str).str.contains('LMS', case=False, na=False)
result = df[mask_p1 & mask_lms & human_error_mask]

---------------------------------------
### LAST 30 DAYS LOGIC EXAMPLE
If the query mentions "last 30 days" and there is a column named last_30_days:

mask_p1 = (df['Priority'] == 'P1')
mask_last_30 = (df['last_30_days'] == 1)
result = df[mask_p1 & mask_last_30]

---------------------------------------
If the query mentions "Major Code Release" and there is a column named Comments:

mask_comments = df['Comments'].astype(str).str.contains('Major Code Release', case=False, na=False)
result = df[mask_comments]

----------------------------------------
If the query mentions "Repeat" and there is a column named Repeat:

mask_repeat = (df['Repeat'] == 1)
mask_last_24_months = (df['last_24_months'] == 1)
result = df[mask_repeat & mask_last_24_months]
-----------------------------------------------

--------------------------------------------
### INCIDENT NUMBER (INC#) DETECTION LOGIC
When the user query contains an incident number such as "INC0381559", 
you MUST perform an exact match lookup on the column named "INC#".

Use this regex to extract the incident number:
    r"(inc\d+)"  (case-insensitive)

Store the matched value in a variable called:
    inc_number

Then generate the following Pandas mask:

    mask_inc = df['INC#'].astype(str).str.upper() == inc_number.upper()

And the final result:

    result = df[mask_inc]

IMPORTANT RULES:
- ALWAYS convert df['INC#'] to string before comparison.
- ALWAYS output the final dataframe using variable name: result
- Do NOT use 'contains'; use exact match == for INC numbers.
- If no INC number is found, do NOT create mask_inc.

EXAMPLE:
User query: "Show details for INC0381559"

Generated code:

inc_number = "INC0381559"
mask_inc = df['INC#'].astype(str).str.upper() == inc_number.upper()
result = df[mask_inc]
---------------------------------------------

If the query mentions a specific INC# like "INC0381559", use below mentioned code template:
mask_inc = df['INC#'] == 'INC0381559'
inc_0381559 = df[mask_inc]

print(inc_0381559.head())
print(inc_0381559.shape)
---------------------------------------------

### PRB NUMBER (PRBNumber) DETECTION LOGIC
When the user query contains an incident number such as "INC0381559", 
you MUST perform an exact match lookup on the column named "INC#".

Use this regex to extract the incident number:
    r"(prb\d+)"  (case-insensitive)

Store the matched value in a variable called:
    prb_number

Then generate the following Pandas mask:

    mask_prb = df['PRBNumber'].astype(str).str.upper() == prb_number.upper()

And the final result:

    result = df[mask_prb]

IMPORTANT RULES:
- ALWAYS convert df['PRBNumber'] to string before comparison.
- ALWAYS output the final dataframe using variable name: result
- Do NOT use 'contains'; use exact match == for PRB numbers.
- If no PRB number is found, do NOT create mask_prb.

EXAMPLE:
User query: "Show details for PRB0002290"

Generated code:

inc_number = "PRB0002290"
mask_prb = df['PRBNumber'].astype(str).str.upper() == prb_number.upper()
result = df[mask_prb]

----------------------------------------------

If the query mentions "this year" use below mentioned code template:

current_year = df['Date_parsed'].dt.year.max()

mask_year = df['Date_parsed'].dt.year == current_year
mask_p1_lms = (df['Priority'] == 'P1') & (df['Product'] == 'LMS')
mask_repeat = df['Repeat'].astype(str).str.lower().isin(['yes', 'y', 'true', '1'])

p1_lms_repeat_this_year = df[mask_year & mask_p1_lms & mask_repeat]

print(p1_lms_repeat_this_year.head())
print(p1_lms_repeat_this_year.shape)
----------------------------------------------
# If the query mentions "2025" use below mentioned code template:

# # Ensure Date_parsed is datetime
# df['Date_parsed'] = pd.to_datetime(df['Date_parsed'], errors='coerce')

# mask_2025 = df['Date_parsed'].dt.year == 2025
# mask_p1_lms = (df['Priority'] == 'P1') & (df['Product'] == 'LMS')
# mask_repeat = df['Repeat'].astype(str).str.lower().isin(['yes', 'y', 'true', '1'])

# p1_lms_repeat_2025 = df[mask_2025 & mask_p1_lms & mask_repeat]

# print(p1_lms_repeat_2025.head())
# print(p1_lms_repeat_2025.shape)

----------------------------------------------
You are an incident analytics assistant.

When the user enters a query, follow these rules to filter the dataframe:

1. HUMAN ERROR DETECTION  
   If the query contains “human error”, “operator error”, “manual error”, 
   “configuration issue”, “runbook issue”, or “procedure issue”, apply filter:
       mask_human_error = df["Initial Causation"].str.contains("error|manual|misconfig|runbook|procedure", case=False, na=False) 
                          | df["Final Causation"].str.contains(...same patterns...)

2. PRIORITY DETECTION  
   If the query contains P1, P2, or P3:  
       mask_priority = df["Priority"] == "<priority>"

3. RELATED TICKETS DETECTION  
   If the query contains any ticket-like pattern such as:
       LRN-xxxxx
       TIM-xxxxx
       PROV-xxxxx
       CHGxxxxx
       C001xxxxx
   Extract it using regex:
       ticket = regex r"[A-Za-z]+[-]?\d+"
   Then apply:
       mask_ticket = df["Related Tickets"].astype(str).str.contains(ticket, case=False, na=False)

4. Combine detected masks:
       mask = True
       if human error detected: mask &= mask_human_error
       if priority detected:    mask &= mask_priority
       if ticket found:         mask &= mask_ticket

5. Return:
       result = df[mask]

6. If a pattern is not found (e.g., no ticket, no human error), skip that mask.

Always return a pandas DataFrame named 'result'.
-------------------------------------------------
You are an incident analytics assistant.

When the user asks queries like:
- "P1 incidents this year"
- "Human error this year"
- "This year's human error incidents"
- "List human error P1 incidents this year"
- "Human error incidents for this year"

Generate Pandas code that follows *these rules*:

-------------------------------------------------------------------

1. Convert Date_parsed into datetime:

    df['Date_parsed'] = pd.to_datetime(df['Date_parsed'], errors='coerce')

2. Determine the current year dynamically *from the dataset*:

    current_year = df['Date_parsed'].dt.year.max()

3. HUMAN ERROR DETECTION:
   If the query contains "human error", "manual error", "operator error",
   "user error", "misconfig", "misconfiguration", "configuration issue",
   "runbook", "procedure":

       mask_human = (
           df['Initial Causation'].astype(str).str.contains(r"error|manual|misconfig|runbook|procedure|configuration", case=False, na=False)
           |
           df['Final Causation'].astype(str).str.contains(r"error|manual|misconfig|runbook|procedure|configuration", case=False, na=False)
           |
           df['Causation Code'].astype(str).str.contains(r"error|manual|misconfig|runbook|procedure|configuration", case=False, na=False)
       )
   Else:
       mask_human = pd.Series(True, index=df.index)

4. PRIORITY DETECTION:
   If the query contains:
     - "P1" → mask_p1 = df["Priority"] == "P1"
     - "P2" → mask_p2 = df["Priority"] == "P2"
     - "P3" → mask_p3 = df["Priority"] == "P3"
   Else:
       mask_priority = pd.Series(True, index=df.index)

5. "THIS YEAR" FILTER:
   If the query includes "this year":
       mask_year = (df['Date_parsed'].dt.year == current_year)

6. Combine all detected masks:

       result = df[ mask_human & mask_priority & mask_year ]

7. Always return the final dataframe in a variable named:

       result

-------------------------------------------------------------------

Follow these exactly and do not create any variable other than 'result'.


----------------------------------------------
### COLUMN LIST (use only these column names):
{column_list}

----------------------------------------------
You are an expert incident analyst.

User will ask queries like:
"Show P1 LMS repeat incidents in 2025"
"Give me repeat incidents for P1 LMS this year"
"List P1 LMS repeat incidents last year"
"Repeat LMS incidents in March 2024"
"Repeat P1 incidents previous month"

You must:
1. Convert Date_parsed to datetime:
   df['Date_parsed'] = pd.to_datetime(df['Date_parsed'], errors='coerce')

2. Understand year/month language:
   - "this year" → current year
   - "last year" or "previous year" → current year - 1
   - "this month" → current month and year
   - "last month" or "previous month" → (current month - 1), adjusting year
   - If a specific year appears (e.g., 2025), use that
   - If a specific month appears (e.g., January, Jan, 01), use that

3. Construct masks:
   mask_p1_lms = (df["Priority"] == "P1") & (df["Product"] == "LMS")
   mask_repeat = df["Repeat"].astype(str).str.lower().isin(["yes","y","1","true"])

4. Apply year/month filters using:
   df['Date_parsed'].dt.year == <year>
   df['Date_parsed'].dt.month == <month_number>

5. Return:
   result = df[mask_p1_lms & mask_repeat & mask_time]

----------------------------------------------
Return ONLY valid Python code. Do not add comments or explanations.
"""
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0
    )

    return completion.choices[0].message.content.strip()

# ---------------------------------------------------------
# LLM: Explain what the generated code does (high-level)
# ---------------------------------------------------------
def explain_pandas_code(user_query: str, code: str) -> str:
    """
    Get a short explanation of the logic, without exposing internal chain-of-thought.
    """
    system_prompt = """
You are a senior data analyst.
Given a user question and the pandas filtering code that answers it,
explain in simple business language what the filters are doing.

Rules:
- Do NOT show the code in the answer.
- Do NOT describe Python or Pandas syntax.
- Just explain what rows are being selected (e.g., 'P1 incidents in LMS in last 30 days that are human-error related').
- Keep it short (2–4 sentences).
"""
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"USER QUESTION:\n{user_query}\n\nPANDAS CODE:\n{code}"}
        ],
        temperature=0.2
    )
    return completion.choices[0].message.content.strip()

# ---------------------------------------------------------
# Safe execution wrapper
# ---------------------------------------------------------
def execute_pandas_code(code: str, df: pd.DataFrame):
    # Provide only df and pd in the local namespace
    local_vars = {
        "df": df,
        "pd": pd,
    }
    try:
        exec(code, {}, local_vars)
        return local_vars.get("result", None), None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------
# Initialize logs in session state
# ---------------------------------------------------------
if "query_logs" not in st.session_state:
    st.session_state["query_logs"] = []

# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------
st.subheader("💬 Ask a question about your CSV")

default_example = "Human error P1 incidents in LMS in last 30 days"
user_query = st.text_input("Example:", value=default_example)

run_btn = st.button("Run Query")

# ---------------------------------------------------------
# Process Query
# ---------------------------------------------------------
if run_btn and user_query.strip():
    column_list = ", ".join(df.columns)

    st.info("🧠 Generating Pandas code from your question...")
    pandas_code = generate_pandas_code(user_query, column_list)

    st.write("#### Generated Pandas code")
    st.code(pandas_code, language="python")

    # Safety check
    safe, bad_pattern = is_code_safe(pandas_code)
    if not safe:
        error_msg = f"Generated code contains forbidden pattern: `{bad_pattern}`. Aborting execution for safety."
        st.error(error_msg)
        st.session_state["query_logs"].append(
            {
                "query": user_query,
                "code": pandas_code,
                "error": error_msg,
                "rows": None,
            }
        )
    else:
        st.info("▶ Executing Pandas code on DataFrame...")
        result_df, error = execute_pandas_code(pandas_code, df)

        if error:
            st.error(f"Execution Error:\n{error}")
            st.session_state["query_logs"].append(
                {
                    "query": user_query,
                    "code": pandas_code,
                    "error": error,
                    "rows": None,
                }
            )
        elif result_df is None:
            warn_msg = "No result DataFrame named 'result' was created by the generated code."
            st.warning(warn_msg)
            st.session_state["query_logs"].append(
                {
                    "query": user_query,
                    "code": pandas_code,
                    "error": warn_msg,
                    "rows": None,
                }
            )
        else:
            # Automatic detection of missing context
            num_rows = int(result_df.shape[0])
            st.success(f"✅ Query executed successfully — {num_rows} rows returned.")

            st.write("### Results")
            st.dataframe(result_df, use_container_width=True)

            # ----------------- Plotly Visualization -----------------
            import plotly.express as px
            import plotly.graph_objects as go

            with st.expander("📊 Interactive Visualizations (Plotly)"):
                if result_df.empty:
                    st.warning("No data available for visualization.")
                else:
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "📦 By Product",
                        "🚦 By Priority",
                        "🏢 By Data Center",
                        "🧭 By Causation",
                        "📈 Time Series",
                        "🌐 Multi-Dimensional"
                    ])

                    # ------------------------------------------------------------
                    # TAB 1 — BY PRODUCT
                    # ------------------------------------------------------------
                    with tab1:
                        if "Product" in result_df.columns:
                            chart_df = result_df["Product"].value_counts().reset_index()
                            chart_df.columns = ["Product", "Count"]

                            fig = px.bar(
                                chart_df,
                                x="Product",
                                y="Count",
                                color="Product",
                                text="Count",
                                title="Incidents by Product"
                            )
                            fig.update_layout(xaxis_title="Product", yaxis_title="Count")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Column 'Product' not available.")

                    # ------------------------------------------------------------
                    # TAB 2 — BY PRIORITY
                    # ------------------------------------------------------------
                    with tab2:
                        if "Priority" in result_df.columns:
                            chart_df = result_df["Priority"].value_counts().reset_index()
                            chart_df.columns = ["Priority", "Count"]

                            fig = px.pie(
                                chart_df,
                                names="Priority",
                                values="Count",
                                title="Incidents by Priority",
                                hole=0.35
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Column 'Priority' not available.")

                    # ------------------------------------------------------------
                    # TAB 3 — BY DATA CENTER
                    # ------------------------------------------------------------
                    with tab3:
                        if "DC#" in result_df.columns:
                            chart_df = result_df["DC#"].value_counts().reset_index()
                            chart_df.columns = ["DC#", "Count"]

                            fig = px.bar(
                                chart_df,
                                x="DC#",
                                y="Count",
                                color="DC#",
                                text="Count",
                                title="Incidents by Data Center"
                            )
                            fig.update_layout(xaxis_title="Data Center ID", yaxis_title="Count")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Column 'DC#' not available.")

                    # ------------------------------------------------------------
                    # TAB 4 — CAUSATION HIERARCHY
                    # ------------------------------------------------------------
                    with tab4:
                        if {"Final Causation", "Sub Category"}.issubset(result_df.columns):
                            fig = px.treemap(
                                result_df,
                                path=["Final Causation", "Sub Category"],
                                title="Incident Causation Tree (Final → Subcategory)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Causation columns not available.")

                    # ------------------------------------------------------------
                    # TAB 5 — TIME SERIES ANALYSIS
                    # ------------------------------------------------------------
                    with tab5:
                        if "Date_parsed" in result_df.columns:
                            temp_df = result_df.copy()
                            temp_df["Date_parsed"] = pd.to_datetime(temp_df["Date_parsed"], errors="coerce")

                            grouped = (
                                temp_df.groupby(temp_df["Date_parsed"].dt.date)["INC#"]
                                .count()
                                .reset_index()
                            )
                            grouped.columns = ["Date", "Incident Count"]

                            fig = px.line(
                                grouped,
                                x="Date",
                                y="Incident Count",
                                title="Incident Trend Over Time",
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Date_parsed column not available.")

                    # ------------------------------------------------------------
                    # TAB 6 — MULTI-DIMENSIONAL ANALYSIS
                    # ------------------------------------------------------------
                    with tab6:
                        st.markdown("### 🔷 3D Scatter — Product × Priority × Duration")

                        # 3D Scatter
                        if {"Product", "Priority", "Duration(min)"}.issubset(result_df.columns):
                            fig = px.scatter_3d(
                                result_df,
                                x="Product",
                                y="Priority",
                                z="Duration(min)",
                                color="Product",
                                title="3D Scatter — Product vs Priority vs Duration"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Columns needed for 3D scatter not available.")

                        st.markdown("### 🔶 Sunburst — Product → Priority → Final Causation")

                        # Sunburst
                        if {"Product", "Priority", "Final Causation"}.issubset(result_df.columns):
                            fig = px.sunburst(
                                result_df,
                                path=["Product", "Priority", "Final Causation"],
                                title="Sunburst View of Incidents"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Columns required for Sunburst not available.")

                        st.markdown("### 🔷 Box Plot — Duration by Product")

                        # Box plot
                        if {"Product", "Duration(min)"}.issubset(result_df.columns):
                            fig = px.box(
                                result_df,
                                x="Product",
                                y="Duration(min)",
                                color="Product",
                                title="Duration Distribution by Product"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Duration or Product not available for box plot.")

            # -----------------------------------------------------------

            # Explanation from LLM (what filters did)
            with st.expander("🧾 Explanation of this result (filters used)"):
                try:
                    explanation = explain_pandas_code(user_query, pandas_code)
                    st.write(explanation)
                except Exception as e:
                    st.write(f"(Could not generate explanation: {e})")

            # LangChain summary of the actual data (context-based)
            st.write("### 🧠 LLM Summary of Results")
            try:
                lc_summary = summarize_result_df(result_df, user_query)
                st.write(lc_summary)
            except Exception as e:
                st.error(f"LangChain summarization failed: {e}")
                lc_summary = "Summary generation failed."

            # ⭐ NEW: RCA Analysis
            st.write("### 🧩 RCA Analysis — Root Cause Patterns")
            try:
                rca_output = generate_rca_analysis(result_df)
                st.write(rca_output)
            except Exception as e:
                st.error(f"RCA analysis failed: {e}")

            # Download CSV
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)

            st.download_button(
                "⬇ Download Results CSV",
                csv_buffer.getvalue(),
                "pandas_query_results.csv",
                "text/csv"
            )

            # PDF export (summary + data)
            try:
                pdf_buffer = create_pdf(lc_summary, result_df)
                st.download_button(
                    "📄 Download Summary + Data as PDF",
                    data=pdf_buffer,
                    file_name="incident_summary.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")

            # Log entry
            st.session_state["query_logs"].append(
                {
                    "query": user_query,
                    "code": pandas_code,
                    "error": None,
                    "rows": num_rows,
                }
            )

# ---------------------------------------------------------
# Logs Viewer
# ---------------------------------------------------------
with st.expander("📚 Query & Code Log (this session)"):
    if not st.session_state["query_logs"]:
        st.write("No queries logged yet.")
    else:
        for i, log in enumerate(st.session_state["query_logs"], start=1):
            st.markdown(f"**#{i} – Query:** {log['query']}")
            st.markdown("**Generated code:**")
            st.code(log["code"], language="python")
            if log["error"]:
                st.markdown(f"**Error:** {log['error']}")
            else:
                st.markdown(f"**Rows returned:** {log['rows']}")
            st.markdown("---")
