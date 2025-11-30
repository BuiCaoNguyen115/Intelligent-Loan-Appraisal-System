import streamlit as st
import pandas as pd
import joblib
import numpy as np
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from kneed import KneeLocator  # <--- NEW LIBRARY
import os
import shap

# ==========================================
# 1. CONFIGURATION & STYLING (PROFESSIONAL THEME)
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Intelligent Loan Appraisal System",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# Professional CSS: Banking/Finance Theme
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e9ecef; box-shadow: 2px 0 5px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #0f2c45; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    h1 { border-bottom: 3px solid #1e88e5; padding-bottom: 15px; margin-bottom: 25px; }
    div[data-testid="stMetric"] { background-color: white; padding: 15px 25px; border-radius: 10px; border-left: 5px solid #1e88e5; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
    .stButton > button { background: linear-gradient(to right, #1565c0, #0d47a1); color: white; border: none; border-radius: 6px; font-weight: 600; padding: 0.6rem 1.2rem; transition: all 0.3s ease; }
    .stButton > button:hover { background: linear-gradient(to right, #0d47a1, #1565c0); box-shadow: 0 4px 12px rgba(0,0,0,0.2); transform: translateY(-2px); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #fff; border-radius: 4px 4px 0px 0px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
    .stTabs [aria-selected="true"] { background-color: #e3f2fd; color: #1565c0; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA & MODEL LOADING MANAGER
# ==========================================
MODEL_DIR = 'model_artifacts' 

def get_path(filename):
    return os.path.join(MODEL_DIR, filename)

@st.cache_resource(show_spinner="Loading AI Models & Artifacts...")
def load_system_artifacts():
    if not os.path.exists(MODEL_DIR):
        st.error(f"❌ Critical Error: Model directory '{MODEL_DIR}' not found.")
        st.stop()
        
    return {
        'loan_model': joblib.load(get_path('loan_amount_model.joblib')),
        'interest_model': joblib.load(get_path('month_interest_model.joblib')),
        'kmeans': joblib.load(get_path('kmeans_model.joblib')),
        'scaler': joblib.load(get_path('scaler.joblib')),
        'profiles': joblib.load(get_path('cluster_profiles.joblib')),
        'cat_features': joblib.load(get_path('categorical_features.joblib')),
        'mean_error': joblib.load(get_path('mean_error.joblib')),
        'r2_results': joblib.load(get_path('model_r2_results.joblib')),
        'cat_r2': joblib.load(get_path('catboost_r2_scores.joblib')),
        'opt_metrics': joblib.load(get_path('optimized_catboost_metrics.joblib')),
        'imp_loan': joblib.load(get_path('loan_amount_top_features.joblib')),
        'imp_int': joblib.load(get_path('month_interest_top_features.joblib')),
        'params_loan': joblib.load(get_path('optuna_best_params_loan.joblib')),
        'params_int': joblib.load(get_path('optuna_best_params_interest.joblib')),
        'df_scaled': joblib.load(get_path('df_scaled.joblib')),
        'features_cluster': joblib.load(get_path('clustering_features_list.joblib'))
    }

@st.cache_data(show_spinner="Loading Datasets...")
def load_datasets():
    return {
        'df_original': joblib.load(get_path('df_original_pre_imputation.joblib')),
        'df_imputed': joblib.load(get_path('df_imputed.joblib')),
        'cols_num': joblib.load(get_path('cols_numeric.joblib')),
        'cols_obj': joblib.load(get_path('cols_object.joblib')),
        'df_eda': joblib.load(get_path('df_eda.joblib'))
    }

try:
    models = load_system_artifacts()
    data = load_datasets()
except Exception as e:
    st.error(f"System Initialization Failed: {e}")
    st.stop()

common_features = [
    'PRODUCT_CATEGORY', 'LOAN_TERM', 'LOAN_PURPOSE', 'BUSINESS_LINE',
    'NUMBER_OF_DEPENDANTS', 'PERMANENT_ADDRESS_PROVINCE', 'JOB',
    'COMPANY_ADDRESS_PROVINCE', 'EDUCATION', 'CUSTOMER_INCOME',
    'ACCOMMODATION_TYPE', 'WORKING_IN_YEAR', 'MARITAL_STATUS',
    'INCOME_RESOURCE', 'AGE'
]

# ==========================================
# 3. SIDEBAR: NAVIGATION & STUDENT INFO
# ==========================================
with st.sidebar:
    st.markdown("Intelligent Loan Appraisal System")
    st.markdown("## Loan Appraisal System")
    st.markdown("*AI-Powered Financial Decision Support*")
    st.markdown("---")
    
    menu = {
        "context": "1. Project Context & Data",
        "eda": "2. Exploratory Data Analysis",
        "data_compare": "3. Data Quality Check",
        "cluster": "4. Customer Segmentation",
        "model_perf": "5. Model Performance & AI",
        "dashboard": "6. Prediction Dashboard",
        "bi": "7. Power BI Report"
    }
    selected_page = st.radio("Navigate to:", list(menu.values()), label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 🎓 Project Info")
    st.info("""
    **Student:** Bui Cao Nguyen
    **ID:** GCS210164/001477380
    **Class:** TCS2601
    """)
    st.caption("© 2025 University Graduation Project")

page_key = [k for k, v in menu.items() if v == selected_page][0]
if page_key != "dashboard": st.title(selected_page)

# --- PAGE: PROJECT CONTEXT ---
if page_key == "context":
    st.markdown("### Executive Summary")
    st.markdown("In the modern banking sector, 'Time-to-Yes' and Risk Management are paramount. This project automates loan appraisal using AI.")
    
    data_dict = {
        "Feature Name": ["CONTRACT_NO", "PRODUCT_CATEGORY", "LOAN_TERM", "MONTH_INTEREST", "LOAN_PURPOSE", "BUSINESS_LINE", "LOAN_AMOUNT", "NUMBER_OF_DEPENDANTS", "PERMANENT_ADDRESS_PROVINCE", "JOB", "COMPANY_ADDRESS_PROVINCE", "EDUCATION", "CUSTOMER_INCOME", "ACCOMMODATION_TYPE", "WORKING_IN_YEAR", "MARITAL_STATUS", "INCOME_RESOURCE", "AGE"],
        "Type": ["ID", "Categorical", "Numeric", "Numeric", "Categorical", "Categorical", "Target", "Numeric", "Categorical", "Categorical", "Categorical", "Categorical", "Numeric", "Categorical", "Numeric", "Categorical", "Categorical", "Numeric"],
        "Description": ["ID", "Product Type", "Term (Months)", "Interest Rate", "Purpose", "Business Sector", "Loan Limit (VND)", "Dependants", "Province Code", "Job Title", "Work Province", "Education", "Income", "Housing", "Experience", "Marital Status", "Income Source", "Age"]
    }
    st.dataframe(pd.DataFrame(data_dict), use_container_width=True, hide_index=True)

# --- PAGE: EDA ---
elif page_key == "eda":
    st.markdown("### Exploratory Data Analysis")
    tabs = st.tabs(["Missing Values", "Correlation", "Distributions"])
    with tabs[0]:
        df_missing = data['df_original']
        missing = df_missing.isnull().sum()
        miss_df = pd.DataFrame({'Missing': missing, '%': (missing/len(df_missing))*100}).sort_values('%', ascending=False)
        miss_df = miss_df[miss_df['Missing'] > 0]
        st.dataframe(miss_df.style.format({'%': "{:.2f}%"}))
    with tabs[1]:
        df_corr = data['df_eda'].select_dtypes(include=[np.number]).drop(columns=['CONTRACT_NO'], errors='ignore')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    with tabs[2]:
        # ... (Insight Dictionary Code from previous step) ...
        # Simplified for brevity in this response, paste your Insight Dictionary logic here
        cols = [c for c in data['df_eda'].columns if c != 'CONTRACT_NO']
        col_viz = st.selectbox("Select Feature:", cols)
        fig, ax = plt.subplots(figsize=(10, 4))
        if pd.api.types.is_numeric_dtype(data['df_eda'][col_viz]):
            sns.histplot(data['df_eda'][col_viz], kde=True, ax=ax, color='teal')
        else:
            sns.countplot(y=data['df_eda'][col_viz], ax=ax, palette='viridis', order=data['df_eda'][col_viz].value_counts().index[:10])
        st.pyplot(fig)

# --- PAGE: DATA COMPARE ---
elif page_key == "data_compare":
    st.markdown("### Data Quality Check")
    check_cols = [c for c in (data['cols_num'] + data['cols_obj']) if c != 'CONTRACT_NO' and c in data['df_original'].columns]
    col_view = st.selectbox("Choose column:", check_cols)
    if pd.api.types.is_numeric_dtype(data['df_original'][col_view]):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data['df_original'][col_view].dropna(), kde=True, color='#1565c0', label='Original', ax=ax)
        sns.histplot(data['df_imputed'][col_view], kde=True, color='#c62828', label='Imputed', alpha=0.5, ax=ax)
        ax.legend()
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        s_orig = data['df_original'][col_view].fillna("Missing").astype(str)
        sns.countplot(y=s_orig, ax=ax[0], color='#1565c0', order=s_orig.value_counts().index)
        ax[0].set_title("Original")
        sns.countplot(y=data['df_imputed'][col_view].astype(str), ax=ax[1], color='#c62828', order=data['df_imputed'][col_view].astype(str).value_counts().index)
        ax[1].set_title("Imputed")
        st.pyplot(fig)

# --- PAGE: CUSTOMER SEGMENTATION (UPDATED WITH KNEELOCATOR) ---
elif page_key == "cluster":
    st.markdown("### Customer Segmentation (Advanced)")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("1. Optimal K Determination")
        st.caption("Using **KneeLocator algorithm** to mathematically find the 'elbow' point.")
        
        # Calculate Inertia
        inertia = []
        K_range = range(1, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data['df_imputed'])
            inertia.append(km.inertia_)
            
        # Locate the Knee/Elbow Programmatically
        kl = KneeLocator(K_range, inertia, curve="convex", direction="decreasing")
        optimal_k = kl.elbow
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(K_range, inertia, 'bx-', label='Inertia')
        if optimal_k:
            ax.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red', label=f'Elbow (k={optimal_k})')
            ax.scatter([optimal_k], [inertia[optimal_k-1]], c='red', s=100, zorder=5)
        
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia')
        ax.legend()
        st.pyplot(fig)
        
        if optimal_k:
            st.success(f"Algorithm detected optimal k = {optimal_k}")
        else:
            st.warning("Could not detect a clear elbow point.")

    with col2:
        st.subheader("2. Cluster Profiles")
        st.dataframe(models['profiles'].style.background_gradient(cmap='Greens'), use_container_width=True)

# --- PAGE: MODEL PERFORMANCE (UPDATED WITH RESIDUALS) ---
elif page_key == "model_perf":
    st.markdown("### Comprehensive Model Performance")
    
    # 1. R2 SCORES
    st.subheader("1. Model Benchmarking")
    res = []
    for k, v in models['r2_results'].items():
        t = 'LOAN_AMOUNT' if '_LOAN_AMOUNT' in k else 'MONTH_INTEREST'
        res.append({'Model': k.replace(f'_{t}', ''), 'Target': t, 'R2': v['R2']})
    res.append({'Model': 'CatBoost (Optuna)', 'Target': 'LOAN_AMOUNT', 'R2': models['opt_metrics']['LOAN_AMOUNT_R2']})
    res.append({'Model': 'CatBoost (Optuna)', 'Target': 'MONTH_INTEREST', 'R2': models['opt_metrics']['MONTH_INTEREST_R2']})
    st.dataframe(pd.DataFrame(res).pivot(index='Model', columns='Target', values='R2').style.highlight_max(axis=0), use_container_width=True)
    with st.expander("🛠️ View Optimized Hyperparameters"):
        st.json(models['params_loan'])
    st.markdown("---")
        # 2. FEATURE IMPORTANCE
    st.subheader("2. Best Model Deep Dive (CatBoost + Optuna)")
    st.info("Detailed analysis of the best performing model (CatBoost Optimized).")

    st.write("#### Feature Importance")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.write("**Target: Loan Amount**")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=models['imp_loan'].head(10), palette='Blues_r', ax=ax1)
        st.pyplot(fig1)
    with col_f2:
        st.write("**Target: Interest Rate**")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=models['imp_int'].head(10), palette='Reds_r', ax=ax2)
        st.pyplot(fig2)
    
    st.markdown("---")

    #3. SHAP (Keep previous code for brevity)
        # 3. SHAP ANALYSIS
    st.write("#### Advanced Model Explainability (SHAP Analysis)")
    st.info("SHAP plots explain why the model made a specific prediction by showing the positive/negative impact of each feature.")

    shap_img_path = 'shap_summary.png' 
    if os.path.exists(shap_img_path):
        st.image(shap_img_path, caption="SHAP Summary Plot (Pre-calculated)", use_column_width=True)
    else:
        if st.button("Run SHAP Analysis (Live Calculation)"):
            with st.spinner("Calculating SHAP values..."):
                try:
                    X_sample = data['df_imputed'].head(100)[common_features].copy()
                    for col in models['cat_features']:
                        if col in X_sample.columns:
                            X_sample[col] = X_sample[col].astype(int).astype(str)
                    explainer = shap.TreeExplainer(models['loan_model'])
                    shap_values = explainer.shap_values(X_sample)
                    fig_shap, ax_shap = plt.subplots()
                    shap.summary_plot(shap_values, X_sample, show=False)
                    st.pyplot(plt.gcf()) 
                    plt.clf() 
                except Exception as e:
                    st.error(f"Cannot run SHAP: {e}")
        else:
            st.markdown("*Click button to trigger live calculation.*")



    # 4. RESIDUAL ANALYSIS (NEW SECTION)
    st.subheader("4. Residual Analysis (Error Diagnostics)")
    st.info("Visualizing the difference between Actual vs. Predicted values to check for bias or heteroscedasticity.")
    
    if st.button("Run Residual Diagnostics"):
        with st.spinner("Generating diagnostics on sample data..."):
            # 1. Prepare Sample Data
            sample_df = data['df_imputed'].sample(min(1000, len(data['df_imputed'])), random_state=42)
            
            # 2. Preprocess (Convert cat to string for CatBoost)
            X_sample = sample_df[common_features].copy()
            for c in models['cat_features']:
                if c in X_sample.columns: X_sample[c] = X_sample[c].astype(int).astype(str)
            
            # 3. Predict & Calculate Residuals
            # Loan Amount
            y_true_loan = sample_df['LOAN_AMOUNT']
            y_pred_loan = models['loan_model'].predict(X_sample)
            resid_loan = y_true_loan - y_pred_loan
            
            # Interest
            y_true_int = sample_df['MONTH_INTEREST']
            y_pred_int = models['interest_model'].predict(X_sample)
            resid_int = y_true_int - y_pred_int
            
            # 4. Plotting
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            # Top Left: Loan Amount Residuals
            sns.scatterplot(x=y_pred_loan, y=resid_loan, ax=axs[0,0], alpha=0.5, color='blue')
            axs[0,0].axhline(0, color='red', linestyle='--')
            axs[0,0].set_title('Loan Amount: Residuals vs Predicted')
            axs[0,0].set_xlabel('Predicted Amount')
            axs[0,0].set_ylabel('Residuals')
            
            # Top Right: Loan Amount Distribution
            sns.histplot(resid_loan, kde=True, ax=axs[0,1], color='blue')
            axs[0,1].set_title('Loan Amount: Residual Distribution')
            
            # Bottom Left: Interest Residuals
            sns.scatterplot(x=y_pred_int, y=resid_int, ax=axs[1,0], alpha=0.5, color='green')
            axs[1,0].axhline(0, color='red', linestyle='--')
            axs[1,0].set_title('Interest Rate: Residuals vs Predicted')
            axs[1,0].set_xlabel('Predicted Rate')
            axs[1,0].set_ylabel('Residuals')
            
            # Bottom Right: Interest Distribution
            sns.histplot(resid_int, kde=True, ax=axs[1,1], color='green')
            axs[1,1].set_title('Interest Rate: Residual Distribution')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("**Interpretation:** Ideally, residuals should be randomly scattered around the red line (0) without distinct patterns, and the histogram should look like a bell curve (Normal Distribution).")

# --- PAGE: DASHBOARD ---
elif page_key == "dashboard":
    st.title("Loan Appraisal Dashboard")
    with st.form("input_form"):
        # ... (Same Form Logic as before) ...
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input('Age', 18, 70, 30)
            marital = st.selectbox('Marital Status (Code)', [0, 1, 2, 3])
            edu = st.selectbox('Education (Code)', range(7))
            dependants = st.number_input('Dependants', 0, 10, 0)
        with c2:
            income = st.number_input('Income', 1e6, 5e8, 15e6, step=1e6)
            job = st.selectbox('Job Code', range(10))
            work_year = st.number_input('Work Years', 0, 40, 5)
            income_src = st.selectbox('Income Source', range(6))
        with c3:
            perm_addr = st.number_input('Perm Addr', 0, 100, 0)
            comp_addr = st.number_input('Comp Addr', 0, 100, 0)
            accom_type = st.selectbox('Accom Type', range(6))
        
        c4, c5 = st.columns(2)
        with c4:
            prod_cat = st.selectbox('Product Cat', range(7))
            loan_purp = st.selectbox('Purpose', range(8))
        with c5:
            biz_line = st.selectbox('Biz Line', range(3))
            term = st.slider('Term', 6, 60, 12)
            
        submitted = st.form_submit_button("Run Analysis")

    if submitted:
        # ... (Full Dictionary Logic for Prediction & Clustering from previous fix) ...
        pred_input = pd.DataFrame([{
            'PRODUCT_CATEGORY': prod_cat, 'LOAN_TERM': term, 'LOAN_PURPOSE': loan_purp,
            'BUSINESS_LINE': biz_line, 'NUMBER_OF_DEPENDANTS': dependants,
            'PERMANENT_ADDRESS_PROVINCE': perm_addr, 'JOB': job,
            'COMPANY_ADDRESS_PROVINCE': comp_addr, 'EDUCATION': edu,
            'CUSTOMER_INCOME': income, 'ACCOMMODATION_TYPE': accom_type,
            'WORKING_IN_YEAR': work_year, 'MARITAL_STATUS': marital,
            'INCOME_RESOURCE': income_src, 'AGE': age
        }])[common_features]
        
        for c in models['cat_features']:
            if c in pred_input.columns: pred_input[c] = pred_input[c].astype(int).astype(str)
            
        pred_amt = models['loan_model'].predict(pred_input)[0]
        pred_int = models['interest_model'].predict(pred_input)[0]
        
        full_cluster_dict = {
            'PRODUCT_CATEGORY': prod_cat, 'LOAN_TERM': term, 'LOAN_PURPOSE': loan_purp,
            'BUSINESS_LINE': biz_line, 'NUMBER_OF_DEPENDANTS': dependants,
            'PERMANENT_ADDRESS_PROVINCE': perm_addr, 'JOB': job,
            'COMPANY_ADDRESS_PROVINCE': comp_addr, 'EDUCATION': edu,
            'CUSTOMER_INCOME': income, 'ACCOMMODATION_TYPE': accom_type,
            'WORKING_IN_YEAR': work_year, 'MARITAL_STATUS': marital,
            'INCOME_RESOURCE': income_src, 'AGE': age,
            'LOAN_AMOUNT': pred_amt, 'MONTH_INTEREST': pred_int
        }
        
        cluster_data = pd.DataFrame([full_cluster_dict])[models['features_cluster']]
        cluster_id = models['kmeans'].predict(models['scaler'].transform(cluster_data))[0]
        
        st.success("Done!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Loan Limit", f"{pred_amt:,.0f}")
        m2.metric("Interest Rate", f"{pred_int:.2f}%")
        m3.metric("Segment", f"Cluster {cluster_id}")
        cluster_names = {0: "Potential VIP", 1: "Stable Customer", 2: "Mass Market"}
        st.info(f"**Segment Characteristics:** {cluster_names.get(cluster_id)}")
# --- PAGE: BI ---
elif page_key == "bi":
    st.title("Power BI")
    components.html("""<iframe title="Report" width="100%" height="800" src="https://app.powerbi.com/view?r=eyJrIjoiN2NjZTY0YTQtODhkMy00NzkwLWExZGQtYzM3OGZmN2NkZjhkIiwidCI6IjA4MzZmY2IyLWJlNzktNGQ5Ny05YTkzLTFiMzRhZWJhNzdiZSIsImMiOjEwfQ%3D%3D" frameborder="0" allowFullScreen="true"></iframe>""", height=800)
