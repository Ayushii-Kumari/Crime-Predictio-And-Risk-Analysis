import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.metrics import classification_report, accuracy_score 
import streamlit as st 
import joblib  


crime_data = pd.read_csv('districtwise-ipc-crimes-2017-onwards.csv')
crime_grouped_data = crime_data.groupby(['state_name', 'district_name', 'year']).agg({
    'murder': 'sum',
    'hit_and_run': 'sum',
    'cheating_impersonation': 'sum',
    'arson': 'sum',
    'criminal_trespass': 'sum',
    'crlty_husbnd_relatives': 'sum',
    'criminal_intimidation': 'sum',
}).reset_index()

crime_grouped_data['crime_count'] = crime_grouped_data[
    ['murder', 'hit_and_run', 'cheating_impersonation', 'arson', 
     'criminal_trespass', 'crlty_husbnd_relatives', 'criminal_intimidation']
].sum(axis=1)

def get_crime_count(year, state_name, district_name):
    
    filtered_data = crime_grouped_data[(crime_grouped_data['year'] == year) &
                                       (crime_grouped_data['state_name'] == state_name) &
                                       (crime_grouped_data['district_name'] == district_name)]
    
    if not filtered_data.empty:
        return filtered_data['crime_count'].mean()
    else:
        return 0 


crime_improvised_data = pd.melt(
    crime_grouped_data,
    id_vars=['state_name', 'district_name', 'year'],
    value_vars=['murder', 'hit_and_run', 'cheating_impersonation', 'arson', 
                'criminal_trespass', 'crlty_husbnd_relatives', 'criminal_intimidation'],
    var_name='crime_type',
    value_name='crime_total_count'  
)

crime_improvised_data = crime_improvised_data[crime_improvised_data['crime_total_count'] > 0]
threshold_value = crime_improvised_data['crime_total_count'].quantile(0.53)
crime_improvised_data['risk_level'] = crime_improvised_data['crime_total_count'].apply(
    lambda x: 'High Risk' if x >= threshold_value else 'Low Risk'
)

socioeconomic_data = pd.read_csv('districtwise-ipc-crimes-2017-onwards.csv')  # Replace with actual path if using
crime_rate = pd.merge(crime_improvised_data, socioeconomic_data, on=['state_name', 'district_name'], how='left')

print("Missing Values Before Handling:")
print(crime_rate.isnull().sum())

numerical_cols = ['crime_total_count']  
crime_rate[numerical_cols] = crime_rate[numerical_cols].fillna(crime_rate[numerical_cols].mean())
categorical_cols = ['state_name', 'district_name']  
for col in categorical_cols:
    crime_rate[col] = crime_rate[col].fillna(crime_rate[col].mode()[0])

print("\nMissing Values After Handling:")
print(crime_rate.isnull().sum())

feature_cols = ['crime_total_count', 'state_name', 'district_name']  
X = crime_rate[feature_cols]
y = crime_rate[['crime_type', 'risk_level']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training set size: {X_train.shape}')
print(f'Test set size: {X_test.shape}')


le_district = LabelEncoder()
X_train['district_name'] = le_district.fit_transform(X_train['district_name'])
X_test['district_name'] = le_district.transform(X_test['district_name'])

le_state = LabelEncoder()
X_train['state_name'] = le_state.fit_transform(X_train['state_name'])
X_test['state_name'] = le_state.transform(X_test['state_name'])

le_crime = LabelEncoder()
le_risk = LabelEncoder()
y_train_crime = le_crime.fit_transform(y_train['crime_type'])
y_test_crime = le_crime.transform(y_test['crime_type'])
y_train_risk = le_risk.fit_transform(y_train['risk_level'])
y_test_risk = le_risk.transform(y_test['risk_level'])

y_train_encoded = np.column_stack((y_train_crime, y_train_risk))
y_test_encoded = np.column_stack((y_test_crime, y_test_risk))


random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
multi_output_model = MultiOutputClassifier(random_forest, n_jobs=-1)
multi_output_model.fit(X_train, y_train_encoded)

y_pred = multi_output_model.predict(X_test)

crime_accuracy = accuracy_score(y_test_encoded[:, 0], y_pred[:, 0])
risk_accuracy = accuracy_score(y_test_encoded[:, 1], y_pred[:, 1])
print(f'Crime Type Prediction Accuracy: {crime_accuracy:.2f}')
print(f'Risk Level Prediction Accuracy: {risk_accuracy:.2f}')

print("\nCrime Type Classification Report:")
print(classification_report(y_test_encoded[:, 0], y_pred[:, 0]))

print("\nRisk Level Classification Report:")
print(classification_report(y_test_encoded[:, 1], y_pred[:, 1]))


@st.cache_data
def load_data():
    data = pd.read_csv('districtwise-ipc-crimes-2017-onwards.csv')
    return data


joblib.dump(multi_output_model, "multi_output_model.pkl")
joblib.dump(le_crime, "le_crime.pkl")
joblib.dump(le_risk, "le_risk.pkl")

model = joblib.load("multi_output_model.pkl")
le_crime = joblib.load("le_crime.pkl")  
le_risk = joblib.load("le_risk.pkl")    

crime_data = pd.read_csv("districtwise-ipc-crimes-2017-onwards.csv") 

years = list(range(2010, 2031)) 
states = sorted(crime_data['state_name'].unique())

def get_districts(state_name):
    return sorted(crime_data[crime_data['state_name'] == state_name]['district_name'].unique())

def get_crime_count(year, state_name, district_name):
    
    filtered_data = crime_grouped_data[(crime_grouped_data['year'] == year) &
                                       (crime_grouped_data['state_name'] == state_name) &
                                       (crime_grouped_data['district_name'] == district_name)]
    
def predict_crime(year, state_name, district_name):
    
    crime_count = get_crime_count(year, state_name, district_name)

    input_data = pd.DataFrame([[crime_count, state_name, district_name]], 
                              columns=['crime_total_count', 'state_name', 'district_name'])

    input_data['state_name'] = le_state.transform(input_data['state_name'])
    input_data['district_name'] = le_district.transform(input_data['district_name'])

    prediction = model.predict(input_data)
    crime_type = le_crime.inverse_transform([prediction[0][0]])[0]  
    risk_level = le_risk.inverse_transform([prediction[0][1]])[0]   

    return crime_type, risk_level


def main():
    st.title("Crime Prediction and Risk Analysis")
    st.write("Select the inputs for prediction:")

    year = st.selectbox("Select Year:", years)

    state_name = st.selectbox("Select State:", states)

    districts = get_districts(state_name)
    district_name = st.selectbox("Select District:", districts)

    if st.button("Predict"):
        crime_type, risk_level = predict_crime(year, state_name, district_name)
        st.write(f"Predicted Major Crime Type: {crime_type}")
        st.write(f"Predicted Risk Level: {risk_level}")

if __name__ == "__main__":
    main()
