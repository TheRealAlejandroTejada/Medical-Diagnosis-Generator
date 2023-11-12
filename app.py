import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template

#Förbehandling av Data
diabetes_data = pd.read_csv('diabetes.csv')
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Dela in i tränings- och testuppsättning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisering
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Funktion för Att Välja Funktioner
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFECV(model, step=1, cv=5, min_features_to_select=8)
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
selected_features = X.columns[rfe.support_]

# Utveckling av Maskininlärningsmodeller
model.fit(X_train_scaled, y_train)

# Validering och Utvärdering av Modellen
X_test_rfe = X_test_scaled[:, rfe.support_]
accuracy = model.score(X_test_scaled, y_test)
print(f'Modellprecision: {accuracy:.2f}')

#  Användargränssnitt (Flask)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Hämta data från formuläret
    data = [float(x) for x in request.form.values()]
    # Förbehandla data
    data_scaled = scaler.transform([data])
    # Välj relevanta funktioner
    data_rfe = data_scaled[:, rfe.support_]
    # Gör förutsägelsen
    prediction = model.predict(data_rfe)

    # Generering av Personlig Diagnos
    if prediction[0] == 1:
        diagnosis = "Positivt för diabetes. Vidare utvärdering rekommenderas."
        lifestyle_recommendations = "Livsstilsrekommendationer för patienter med diabetes: Behåll en balanserad kost, träna regelbundet och följ upp regelbundet med en vårdgivare."
        prevention_recommendations = "Förebyggande åtgärder: Regelbunden övervakning av glukosnivåer, periodisk konsultation med en endokrinolog och följa medicinska rekommendationer."
    else:
        diagnosis = "Negativt för diabetes. Inga tecken på sjukdomen observerades."
        lifestyle_recommendations = "Allmänna livsstilsrekommendationer: Behåll en hälsosam kost, träna regelbundet och schemalägg regelbundna hälsoundersökningar."
        prevention_recommendations = "Allmänna förebyggande åtgärder: Behåll en hälsosam livsstil, undvik överdrivet intag av socker och behåll en hälsosam kroppsvikt."

    return render_template('result.html', prediction=prediction[0], diagnosis=diagnosis, lifestyle_recommendations=lifestyle_recommendations, prevention_recommendations=prevention_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
