import plotly.graph_objs as go
import plotly.io as pio
from datetime import datetime

# Sample data (replace this with your actual data)
person_data = {
    'Person 1': [(datetime(2024, 5, 1, 12, 0), 10), (datetime(2024, 5, 1, 12, 30), 15), (datetime(2024, 5, 1, 13, 0), 12)],
    'Person 2': [(datetime(2024, 5, 1, 12, 0), 8), (datetime(2024, 5, 1, 12, 30), 10), (datetime(2024, 5, 1, 13, 0), 11)]
    # Add data for other persons similarly
}

# Extract x-axis (timestamps) and y-axis (data) for each person
x_data = {person: [point[0] for point in data] for person, data in person_data.items()}
y_data = {person: [point[1] for point in data] for person, data in person_data.items()}

# Create traces for each person
traces = []
for person in person_data:
    trace = go.Scatter(x=x_data[person], y=y_data[person], mode='lines+markers', name=person)
    traces.append(trace)

# Layout
layout = go.Layout(title='Data from Video Cams', xaxis=dict(title='Timestamp'), yaxis=dict(title='Data'))

# Create figure
fig = go.Figure(data=traces, layout=layout)

# Plot
pio.show(fig)
