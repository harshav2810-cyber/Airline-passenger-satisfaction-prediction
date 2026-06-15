import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("airline_satisfaction_model.pkl")

st.set_page_config(
    page_title="✈️ Airline Passenger Satisfaction Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Airline Passenger Satisfaction Predictor")

st.markdown("""
Welcome! This application predicts whether an airline passenger is likely to be **Satisfied**
or **Neutral / Dissatisfied** based on travel information and service ratings.

### ⭐ Rating Guide
- **0** = Very Poor
- **1** = Poor
- **2** = Fair
- **3** = Good
- **4** = Very Good
- **5** = Excellent
""")

st.markdown("---")

# -----------------------
# User Inputs
# -----------------------

gender_option = st.selectbox("Gender", ["Female", "Male"])

# Label Encoding mapping:
# Female -> 0, Male -> 1
gender = 0 if gender_option == "Female" else 1

age = st.number_input("Age", min_value=1, max_value=100, value=30)

customer_option = st.selectbox(
    "Customer Type",
    ["Disloyal Customer", "Loyal Customer"]
)

# Disloyal -> 0, Loyal -> 1
customer_type = 0 if customer_option == "Disloyal Customer" else 1

travel_option = st.selectbox(
    "Type of Travel",
    ["Business Travel", "Personal Travel"]
)

# Business Travel -> 0, Personal Travel -> 1
type_of_travel = 0 if travel_option == "Business Travel" else 1

flight_distance = st.number_input("Flight Distance", min_value=0, value=500)

departure_delay = st.number_input("Departure Delay", min_value=0, value=0)
arrival_delay = st.number_input("Arrival Delay", min_value=0, value=0)

st.subheader("⭐ Service Ratings")

departure_and_arrival_time_convenience = st.slider(
    "Departure & Arrival Time Convenience", 0, 5, 3
)

ease_of_online_booking = st.slider(
    "Ease of Online Booking", 0, 5, 3
)

check_in_service = st.slider(
    "Check-in Service", 0, 5, 3
)

online_boarding = st.slider(
    "Online Boarding", 0, 5, 3
)

gate_location = st.slider(
    "Gate Location", 0, 5, 3
)

on_board_service = st.slider(
    "On-board Service", 0, 5, 3
)

seat_comfort = st.slider(
    "Seat Comfort", 0, 5, 3
)

leg_room_service = st.slider(
    "Leg Room Service", 0, 5, 3
)

cleanliness = st.slider(
    "Cleanliness", 0, 5, 3
)

food_and_drink = st.slider(
    "Food and Drink", 0, 5, 3
)

in_flight_service = st.slider(
    "In-flight Service", 0, 5, 3
)

in_flight_wifi_service = st.slider(
    "In-flight WiFi Service", 0, 5, 3
)

in_flight_entertainment = st.slider(
    "In-flight Entertainment", 0, 5, 3
)

baggage_handling = st.slider(
    "Baggage Handling", 0, 5, 3
)

travel_class = st.selectbox(
    "Travel Class",
    ["Business", "Economy", "Economy Plus"]
)

# -----------------------
# One-Hot Encoding
# -----------------------

class_business = 1 if travel_class == "Business" else 0
class_economy = 1 if travel_class == "Economy" else 0
class_economy_plus = 1 if travel_class == "Economy Plus" else 0

# -----------------------
# Create Input DataFrame
# -----------------------

input_df = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "customer_type": [customer_type],
    "type_of_travel": [type_of_travel],
    "flight_distance": [flight_distance],
    "departure_delay": [departure_delay],
    "arrival_delay": [arrival_delay],
    "departure_and_arrival_time_convenience": [departure_and_arrival_time_convenience],
    "ease_of_online_booking": [ease_of_online_booking],
    "check-in_service": [check_in_service],
    "online_boarding": [online_boarding],
    "gate_location": [gate_location],
    "on-board_service": [on_board_service],
    "seat_comfort": [seat_comfort],
    "leg_room_service": [leg_room_service],
    "cleanliness": [cleanliness],
    "food_and_drink": [food_and_drink],
    "in-flight_service": [in_flight_service],
    "in-flight_wifi_service": [in_flight_wifi_service],
    "in-flight_entertainment": [in_flight_entertainment],
    "baggage_handling": [baggage_handling],
    "class_Business": [class_business],
    "class_Economy": [class_economy],
    "class_Economy Plus": [class_economy_plus],
})

# -----------------------
# Prediction
# -----------------------

if st.button("Predict Satisfaction"):

    prediction = model.predict(input_df)

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:
        st.success(
            "✅ The passenger is likely to be SATISFIED with the airline experience."
        )
    else:
        st.error(
            "⚠️ The passenger is likely to be NEUTRAL OR DISSATISFIED with the airline experience."
        )


st.markdown("---")
st.caption(
    "Developed by Harsha Vardhan Reddy | Machine Learning Project using Streamlit & Random Forest"
)

