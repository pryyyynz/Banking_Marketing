import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


class BankMarketingPreprocessor:
    """
    A class to handle preprocessing for the Bank Marketing dataset
    """

    def __init__(self):
        # Define mappings and transformations based on the original preprocessing

        # Binary categorical variables mapping
        self.binary_mapping = {
            'default': {'no': 0, 'yes': 1, 'unknown': 0},
            'housing': {'no': 0, 'yes': 1, 'unknown': 0},
            'loan': {'no': 0, 'yes': 1, 'unknown': 0},
        }

        # Education order for ordinal encoding
        self.education_order = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
                                'high.school', 'professional.course', 'university.degree', 'unknown']

        # Month mapping for cyclical encoding
        self.month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

        # Day of week mapping for cyclical encoding
        self.day_order = {'mon': 1, 'tue': 2, 'wed': 3,
                          'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}

        # Features to be one-hot encoded
        self.nominal_cols = ['job', 'marital', 'contact', 'poutcome']

        # Feature list based on encoded_normalized dataset
        self.encoded_data_sample = pd.read_csv(
            '../bank_encoded_normalized.csv', nrows=1)
        self.feature_list = [
            col for col in self.encoded_data_sample.columns if col != 'y']

        # Initialize encoders
        self.education_encoder = OrdinalEncoder(
            categories=[self.education_order])

        # Initialize scaler for numerical features
        self.scaler = StandardScaler()

        # Train encoders and scaler on a sample if needed
        # For a real implementation, these should be loaded from saved preprocessing objects

    def preprocess_individual(self, data_dict):
        """
        Preprocess a single data point
        """
        # Create a DataFrame with the correct feature columns
        processed = pd.DataFrame(
            np.zeros((1, len(self.feature_list))), columns=self.feature_list)

        # 1. Handle binary encoding
        for col in self.binary_mapping:
            if col in data_dict:
                processed[col] = self.binary_mapping[col].get(
                    data_dict[col], 0)

        # 2. Handle education ordinal encoding
        if 'education' in data_dict:
            education_value = data_dict['education']
            education_idx = self.education_order.index(
                education_value) if education_value in self.education_order else 0
            processed['education_encoded'] = education_idx

        # 3. Handle cyclical encoding for month and day
        if 'month' in data_dict:
            month_num = self.month_order.get(data_dict['month'], 1)
            processed['month_sin'] = np.sin(2 * np.pi * month_num / 12)
            processed['month_cos'] = np.cos(2 * np.pi * month_num / 12)

        if 'day_of_week' in data_dict:
            day_num = self.day_order.get(data_dict['day_of_week'], 1)
            processed['day_sin'] = np.sin(2 * np.pi * day_num / 7)
            processed['day_cos'] = np.cos(2 * np.pi * day_num / 7)

        # 4. Handle one-hot encoding for nominal columns
        # Apply manual one-hot encoding for job
        if 'job' in data_dict:
            job_value = data_dict['job']
            job_columns = [
                col for col in self.feature_list if col.startswith('job_')]
            for col in job_columns:
                job_category = col.replace('job_', '')
                processed[col] = 1 if job_category == job_value else 0

        # Apply manual one-hot encoding for marital
        if 'marital' in data_dict:
            marital_value = data_dict['marital']
            marital_columns = [
                col for col in self.feature_list if col.startswith('marital_')]
            for col in marital_columns:
                marital_category = col.replace('marital_', '')
                processed[col] = 1 if marital_category == marital_value else 0

        # Apply manual one-hot encoding for contact
        if 'contact' in data_dict:
            contact_value = data_dict['contact']
            contact_columns = [
                col for col in self.feature_list if col.startswith('contact_')]
            for col in contact_columns:
                contact_category = col.replace('contact_', '')
                processed[col] = 1 if contact_category == contact_value else 0

        # Apply manual one-hot encoding for poutcome
        if 'poutcome' in data_dict:
            poutcome_value = data_dict['poutcome']
            poutcome_columns = [
                col for col in self.feature_list if col.startswith('poutcome_')]
            for col in poutcome_columns:
                poutcome_category = col.replace('poutcome_', '')
                processed[col] = 1 if poutcome_category == poutcome_value else 0

        # 5. Handle numerical features and normalization
        numerical_features = ['age', 'campaign', 'pdays', 'previous',
                              'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                              'euribor3m', 'nr.employed']

        for col in numerical_features:
            if col in data_dict:
                # In a real implementation, we would apply proper scaling
                # Here we just use the raw value as a placeholder
                column_name = col
                if column_name in processed.columns:
                    processed[column_name] = data_dict[col]

        # 6. Create derived features
        # Campaign efficiency ratio
        if 'campaign' in data_dict and 'previous' in data_dict:
            processed['campaign_previous_ratio'] = data_dict['campaign'] / \
                (data_dict['previous'] + 1)

        # Economic sentiment
        if all(col in data_dict for col in ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'nr.employed']):
            processed['economic_sentiment'] = (
                data_dict['emp.var.rate'] +
                data_dict['cons.conf.idx'] +
                data_dict['nr.employed'] -
                abs(data_dict['cons.price.idx'])
            )

        # Has previous contact
        if 'pdays' in data_dict:
            processed['has_previous_contact'] = 0 if data_dict['pdays'] == 999 else 1

        # Age group features
        if 'age' in data_dict:
            age = data_dict['age']
            age_group_cols = [
                col for col in self.feature_list if col.startswith('age_group_')]
            for col in age_group_cols:
                group = col.replace('age_group_', '')

                if group == '18-30' and 18 <= age <= 30:
                    processed[col] = 1
                elif group == '31-40' and 31 <= age <= 40:
                    processed[col] = 1
                elif group == '41-50' and 41 <= age <= 50:
                    processed[col] = 1
                elif group == '51-60' and 51 <= age <= 60:
                    processed[col] = 1
                elif group == '60+' and age > 60:
                    processed[col] = 1
                else:
                    processed[col] = 0

        return processed

    def preprocess_batch(self, df):
        """
        Preprocess a batch of data
        """
        # Create a DataFrame with the correct feature columns
        processed = pd.DataFrame(
            np.zeros((len(df), len(self.feature_list))), columns=self.feature_list)

        # Process each row individually (not efficient, but reliable)
        for idx, row in df.iterrows():
            data_dict = row.to_dict()
            processed.iloc[idx] = self.preprocess_individual(data_dict).iloc[0]

        return processed
