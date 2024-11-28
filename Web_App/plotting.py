import streamlit as st

# Set the title of the app
st.title("AgriPredict")

# Create a box container for the buttons and dropdown
with st.container():
    # Title for the button section
    st.subheader("Select Time Period")

    # Create 5 columns to arrange buttons in a line
    col1, col2, col3, col4, col5 = st.columns(5)

    # Define the button labels and actions
    with col1:
        if st.button('2 Weeks'):
            st.session_state.selected_period = '2 Weeks'
            st.write("You selected 2 Weeks")

    with col2:
        if st.button('1 Month'):
            st.session_state.selected_period = '1 Month'
            st.write("You selected 1 Month")

    with col3:
        if st.button('3 Months'):
            st.session_state.selected_period = '3 Months'
            st.write("You selected 3 Months")

    with col4:
        if st.button('6 Months'):
            st.session_state.selected_period = '6 Months'
            st.write("You selected 6 Months")

    with col5:
        if st.button('1 Year'):
            st.session_state.selected_period = '1 Year'
            st.write("You selected 1 Year")

# Dropdown to select the state
st.subheader("Select State")
states = ["Karnataka", "Uttar Pradesh", "Madhya Pradesh", "Telangana", "Gujarat"]
selected_state = st.selectbox("Choose a state", states)

# Display the selected state
st.write(f"You selected: {selected_state}")

# Add a placeholder for the graph later (optional for now)
st.write("Graph will appear here based on the selected time period and state.")
