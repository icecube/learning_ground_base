
# Function to update the right dropdown based on left dropdown selection
def update_step_dropdown(state):
    state.selected_right = state.log_structure[state.selected_left]["steps"][0]
