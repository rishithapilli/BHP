# FROM python:3.12.0-slim
# COPY . /app
# WORKDIR /app
# # RUN pip install -r requirements.txt
# # CMD ["python","app.py"]
# # Copy requirements.txt (if you have one) and install dependencies
# COPY requirements.txt ./
# RUN pip install -r requirements.txt

# # Copy your application code, notebooks, and static files
# COPY . .

# # Set the environment variable for Flask app entry point (optional)
# ENV FLASK_APP=app.py  

# # Expose the port where Flask listens (usually 5000)
# EXPOSE 5000

# # Run the Flask app using gunicorn (recommended for production)
# CMD ["python","app.py"]



FROM python:3.12-slim
 
# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn
    

# Copy your application code, notebooks, and static files
COPY . .

# Set the environment variable for Flask app entry point (optional)
ENV FLASK_APP=app.py  

# Expose the port where Flask listens (usually 5000)
EXPOSE 5000

# Run the Flask app using gunicorn (recommended for production)
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

CMD ["python", "app.py"]