
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY src/serve/docker/requirements.txt ./

# Copy the application code into the container
COPY . ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model files into the container
COPY src/models/model.onnx/onnx/ ./model/

# Expose port 5000 to the outside world
EXPOSE 5000

# Run the application
CMD ["python", "src/serve/docker/serviceZadMong.py"]