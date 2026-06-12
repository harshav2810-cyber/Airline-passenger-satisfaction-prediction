import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load files
model = joblib.load("airline_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("✈️ Airline Passenger Satisfaction Prediction System")

st.markdown("""
This machine learning application predicts whether a passenger is likely to be satisfied with an airline experience based on service quality, travel details, and customer information.
""")

st.write("Enter passenger details below")

# Numerical Inputs
age = st.number_input("Age", 1, 100, 25)

flight_distance = st.number_input(
    "Flight Distance", 0, 10000, 500
)

departure_delay = st.number_input(
    "Departure Delay", 0, 2000, 0
)

arrival_delay = st.number_input(
    "Arrival Delay", 0, 2000, 0
)

# Service Ratings
departure_arrival_convenience = st.slider(
    "Departure and Arrival Time Convenience",
    0, 5, 3
)

ease_online_booking = st.slider(
    "Ease of Online Booking",
    0, 5, 3
)

checkin_service = st.slider(
    "Check-in Service",
    0, 5, 3
)

online_boarding = st.slider(
    "Online Boarding",
    0, 5, 3
)

gate_location = st.slider(
    "Gate Location",
    0, 5, 3
)

onboard_service = st.slider(
    "On-board Service",
    0, 5, 3
)

seat_comfort = st.slider(
    "Seat Comfort",
    0, 5, 3
)

leg_room_service = st.slider(
    "Leg Room Service",
    0, 5, 3
)

cleanliness = st.slider(
    "Cleanliness",
    0, 5, 3
)

food_drink = st.slider(
    "Food and Drink",
    0, 5, 3
)

inflight_service = st.slider(
    "In-flight Service",
    0, 5, 3
)

wifi = st.slider(
    "In-flight Wifi Service",
    0, 5, 3
)

entertainment = st.slider(
    "In-flight Entertainment",
    0, 5, 3
)

baggage = st.slider(
    "Baggage Handling",
    0, 5, 3
)

# Categorical Inputs

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

customer_type = st.selectbox(
    "Customer Type",
    ["Loyal Customer", "Returning"]
)

travel_type = st.selectbox(
    "Type of Travel",
    ["Business", "Personal"]
)

travel_class = st.selectbox(
    "Class",
    ["Business", "Economy", "Economy Plus"]
)

if st.button("Predict Satisfaction"):

    gender_male = 1 if gender == "Male" else 0

    customer_returning = (
        1 if customer_type == "Returning" else 0
    )

    personal_travel = (
        1 if travel_type == "Personal" else 0
    )

    class_economy = (
        1 if travel_class == "Economy" else 0
    )

    class_economy_plus = (
        1 if travel_class == "Economy Plus" else 0
    )

    data = pd.DataFrame([[
        age,
        flight_distance,
        departure_delay,
        arrival_delay,
        departure_arrival_convenience,
        ease_online_booking,
        checkin_service,
        online_boarding,
        gate_location,
        onboard_service,
        seat_comfort,
        leg_room_service,
        cleanliness,
        food_drink,
        inflight_service,
        wifi,
        entertainment,
        baggage,
        gender_male,
        customer_returning,
        personal_travel,
        class_economy,
        class_economy_plus
    ]],
    columns=[
        'Age',
        'Flight Distance',
        'Departure Delay',
        'Arrival Delay',
        'Departure and Arrival Time Convenience',
        'Ease of Online Booking',
        'Check-in Service',
        'Online Boarding',
        'Gate Location',
        'On-board Service',
        'Seat Comfort',
        'Leg Room Service',
        'Cleanliness',
        'Food and Drink',
        'In-flight Service',
        'In-flight Wifi Service',
        'In-flight Entertainment',
        'Baggage Handling',
        'Gender_Male',
        'Customer Type_Returning',
        'Type of Travel_Personal',
        'Class_Economy',
        'Class_Economy Plus'
    ])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.success(
            "Passenger is likely SATISFIED 😊"
        )
    else:
        st.error(
            "Passenger is likely NEUTRAL / DISSATISFIED 😐"
        )