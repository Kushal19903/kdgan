#!/bin/bash

# Start the API server
cd code
python api.py &
API_PID=$!
cd ..

# Start the frontend
cd kdgan-frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Function to handle termination
function cleanup {
  echo "Stopping servers..."
  kill $API_PID
  kill $FRONTEND_PID
  exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

echo "API server running on http://localhost:5000"
echo "Frontend running on http://localhost:3000"
echo "Press Ctrl+C to stop both servers"

# Wait for user to press Ctrl+C
wait