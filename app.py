import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from feature_engineering.missing_imputation import Imputer
from feature_engineering.date_handling import DateTimeHandler
from feature_engineering.encode import Encoder
from feature_engineering.outlier_removing import OutlierRemover
from feature_engineering.feature_selection import FeatureSelector
from feature_engineering.scaling import Scaler
from feature_engineering.pipeline import CustomPipeline
from feature_engineering.pipeline import notebook_generator
from sklearn.preprocessing import LabelEncoder
import base64
from feature_engineering.pipeline import DatasetCleaner
import time
from PIL import Image
from utils.helper import log_dictionary

# Specify the desired logo size
logo_size = 15
logo = Image.open('logo/2.png')
st.image(logo, width=logo_size, use_column_width='auto', caption='', output_format='JPEG')

st.set_option('deprecation.showPyplotGlobalUse', False)

MAX_FILE_SIZE = 15 * 1024 * 1024
data_source = st.selectbox("Select Data Source:", ["Upload a CSV File", "Use Titanic Dataset"])

if data_source == "Upload a CSV File":
    uploaded_file = st.file_uploader("Beta version", type=["csv"])

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin experiments. (Max. 15 mb)")
        st.stop()

    if uploaded_file is not None:

        if len(uploaded_file.read()) > MAX_FILE_SIZE:
            st.error(f"File size exceeds the maximum allowed size of 15 MB.")
            st.stop()
        else:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            cols = list(df.columns)
            cols.insert(0, '')
            task_type = st.selectbox('Your problem is: ', ['','classification', 'regression'])
            target = st.selectbox('Your target column is: ', cols)

elif data_source == "Use Titanic Dataset":

    df = pd.read_csv("data/titanic.csv")
    task_type = st.selectbox('Your problem is: ', ['classification'])
    target = st.selectbox('Your target column is: ', ['survived'])

if len(task_type)<2 or len(target)<2:

    st.info("Please enter your task type and target column.")
    st.stop()

if 'eda_clicked' not in st.session_state:
    st.session_state.eda_clicked = False

st.markdown("# 1. Automatic EDA 📊")

eda_clicked = st.button("Run EDA ")

# Check if Button 1 is clicked
if eda_clicked:

    dataset_cleaner = DatasetCleaner(df)
    df = dataset_cleaner.clean()
    st.success("""At present, the implementation of automatic Exploratory Data Analysis (EDA) is hindered due to certain library-related challenges. However, we have undertaken preliminary data preparation steps, which include:

        Duplicated Row Removal: We have successfully eliminated duplicated rows from the dataset, ensuring data integrity and accuracy.

        Elimination of Features with Unique Values: Features with singular or unique values have been removed, streamlining the dataset and enhancing its suitability for further analysis.""")
    

st.markdown("# 2. Feature Engineering ⚙️")

with st.form('user_inputs'):

    selected_missing = st.selectbox('Which method do want for Missing Value Imputation?',
                                    ['mean_imputer', 'arbitrary_value_imputer',
                                      'missing_indicator_imputer', 'knn_imputer','iterative_imputer'])
    selected_encoded = st.selectbox('Which method do want for Categorical Encoding?', ['one_hot_encoder',
                                                                                       'ordinal_encoder',
                                                                                       'frequency_encoder',
                                                                                       'mean_encoder',
                                                                                       'ohe_frequent_encoder'])
    selected_outlier = st.selectbox('Which method do want for Outlier removing?',
                                    ['removing_outliers_iqr', 'none','removing_outliers_quantiles'])
    selected_feature = st.selectbox('Which method do want for Feature Selection?',
                                    ['none','select_k_best', 'recursive_feature_elimination'])
    selected_scaling = st.selectbox('Which method do want for Scaling?', ['min_max', 'standard','none', 'robust'])
    submitted = st.form_submit_button("Run experiment 🔬")

if submitted:
    def is_categorical(column):
        return column.dtype == 'object' or pd.api.types.is_categorical_dtype(column)

    if is_categorical(df[target]):
        # Apply label encoding
        label_encoder = LabelEncoder()
        df[target] = label_encoder.fit_transform(df[target])

    X = df.drop(columns=target)
    y = df[target]

    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )

    imputer = Imputer(imputer_type=selected_missing)
    encoder = Encoder(selected_encoded)
    date_handler = DateTimeHandler()
    outlier_remover = OutlierRemover(selected_outlier)
    feature_selector = FeatureSelector(selected_feature)
    scaler = Scaler(scaler_type=selected_scaling)

    steps = [imputer, encoder, date_handler, outlier_remover, feature_selector, scaler]

    pipeline = CustomPipeline(steps, X_train_k, X_test_k, y_train_k, y_test_k)

    X_train_t, X_test_t, y_train, y_test = pipeline.method_transform()

    test_data = pd.concat([X_test_t, y_test], axis=1)
    train_data = pd.concat([X_train_t, y_train], axis=1)

    tasks = [
        "Missing values imputed",
        "Categorical values encoded",
        "Date-related features handled",
        "Outliers removed",
        "Data scaled",
    ]

    progress_container = st.empty()

    for i, task in enumerate(tasks, start=1):
        time.sleep(2)  # Simulate some work
        progress = i / len(tasks)
        progress_container.progress(progress)
        progress_container.text(f"Task {i}/{len(tasks)}: {task}")

    st.markdown("# 3. Model training 🤖")
    with st.spinner("Please wait..."):
        # Create an empty container for the progress bar

        if task_type == 'classification':
            from pycaret.classification import *

            exp = setup(data=train_data,
                        test_data=test_data,
                        target=target,
                        session_id=123)
        elif task_type == 'regression':
            from pycaret.regression import *

            exp = setup(data=train_data,
                        test_data=test_data,
                        target=target,
                        session_id=123)
        st.session_state.best = compare_models()
        st.session_state.df = pull()

        if data_source == "Upload a CSV File":

            notebook_generator(uploaded_file.name, target, selected_missing, selected_encoded, selected_outlier,
                       selected_feature, selected_scaling, st.session_state.best)
        else:
            notebook_generator('titanic.csv', target, selected_missing, selected_encoded, selected_outlier,
                               selected_feature, selected_scaling, st.session_state.best)

        sample_dict = {
            'imputer': selected_missing,
            'encoder': selected_encoded,
            'outlier_remover': selected_outlier,
            'feature_selector': selected_feature,
            'scaler': selected_scaling,
            'model': st.session_state.best
        }

            # Log the dictionary
        log_dictionary(sample_dict)

if "best" in st.session_state:
    st.markdown("### Your best model: 🤩")
    code = f"{st.session_state.best}"
    st.code(code, language="python")

if "df" in st.session_state:
    st.markdown("### Your experiment result: 🎉🎊")
    st.write(st.session_state.df)



if "best" in st.session_state:
    st.markdown("# 5. Model analyzing 🔬")
    # with open('model.pickle', 'rb') as file:
    #     best = pickle.load(file)

    if task_type == 'classification':
        from pycaret.classification import *
        plot_type = st.selectbox('Analyze your model by: ',
                                 ['confusion_matrix', 'error', 'class_report', 'boundary', 'rfe', 'learning',
                                  'manifold', 'calibration', 'vc', 'dimension', 'feature', 'feature_all', 'parameter',
                                  'lift', 'gain', 'tree', 'ks'])
        matrix_plot = plot_model(st.session_state.best, plot=plot_type, display_format='streamlit')

    elif task_type == 'regression':
        from pycaret.regression import *
        plot_type = st.selectbox('Analyze your model by: ', ['residuals', 'error', 'manifold', 'feature_all'])
        matrix_plot = plot_model(st.session_state.best, plot=plot_type, display_format='streamlit')


    st.markdown("# 6. Download notebook ⬇️")

    with open("jupyter_notebook.ipynb", "rb") as notebook_file:

        notebook_content = notebook_file.read()

        b64_notebook = base64.b64encode(notebook_content).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64_notebook}" download="jupyter_notebook.ipynb">Click here to download current experiment as a notebook.</a>'
        st.markdown(href, unsafe_allow_html=True)

    with open('utils/app.log', 'r') as log_file:
        log_data = log_file.read()

        b64_log = base64.b64encode(log_data.encode('utf-8')).decode('utf-8')
        st.markdown(f'<a href="data:text/plain;base64,{b64_log}" download="app.log">Download experiment logs for further analysis.</a>',
                    unsafe_allow_html=True)


st.title("Feedback")
st.markdown("If you have any suggestions, feedback, or encountered any issues with the experimenteer, please [write me.](https://www.linkedin.com/in/nijat-zeynalov-064163142/)")
