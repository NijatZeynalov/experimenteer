# experimenteer

![2](https://github.com/NijatZeynalov/experimenteer/assets/31247506/32bd1480-4724-4690-a0f3-0869468f4bc9)




Automate your classic machine learning experiments with Streamlit-based experimenteer. This tool simplifies the process of dataset preparation, problem type selection (regression or classification), and feature engineering method customization. With just a few clicks, you can create and evaluate machine learning models, conduct multiple experiments, and analyze your results.

## Getting Started

To get started with the AutoML Streamlit application, follow these steps:

1. **Installation**: Make sure you have Python installed on your system. Install the required packages by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Application**: Run the Streamlit application by executing the following command:

   ```bash
   streamlit run app.py
   ```

3. **Upload Your Dataset**: Use the application's interface to upload your dataset.

4. **Select Problem Type**: Choose whether you want to address a regression or classification problem based on your dataset.

5. **Customize Feature Engineering**: Select from a range of feature engineering methods:

   - **Missing Data Imputation**:
     - `mean_imputer`: Fill missing values with the mean of the respective feature.
     - `arbitrary_value_imputer`: Fill missing values with user-defined arbitrary values.
     - `missing_indicator_imputer`: Create binary indicators for missing values.
     - `knn_imputer`: Impute missing values using the k-nearest neighbors algorithm.
     - `iterative_imputer`: Utilize iterative imputation to fill missing data.

   - **Encoding**:
     - `one_hot_encoder`: Perform one-hot encoding for categorical variables.
     - `ordinal_encoder`: Apply ordinal encoding for ordered categorical features.
     - `frequency_encoder`: Encode categorical features based on frequency.
     - `mean_encoder`: Encode categorical features using mean target encoding.
     - `ohe_frequent_encoder`: Combine one-hot encoding with frequent category encoding.

   - **Outlier Removal**:
     - `removing_outliers_iqr`: Remove outliers using the Interquartile Range (IQR) method.
     - `removing_outliers_quantiles`: Remove outliers based on specified quantiles.

   - **Feature Selection**:
     - `select_k_best`: Select the top k features based on statistical tests.
     - `recursive_feature_elimination`: Use recursive feature elimination to select the best features.

   - **Scaling**:
     - `min_max`: Apply Min-Max scaling to normalize features.
     - `standard`: Standardize features to have zero mean and unit variance.
     - `robust`: Scale features robustly to handle outliers.

6. **Experimentation**: Run as many experiments as you like with different feature engineering configurations. In each experiment, the application automatically builds classic machine learning models using your selected feature engineering methods.

7. **Results and Logs**: View the results of each experiment, including performance metrics and model evaluations, directly in the Streamlit app. You can also download a Jupyter notebook containing the entire experiment and logs for reference.

## Example Usage

```bash
# Install required packages
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Special thanks to the open-source community for their contributions to the libraries and tools used in this project.

## Contact

If you have any questions or need assistance, feel free to contact us at [nijatzeynalov98298@gmail.com](mailto:nijatzeynalov98298@gmail.com).

Happy experimenting!
