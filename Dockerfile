FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Make sure the script is executable
RUN chmod +x run_processing.sh

# Use the shell script as the container entrypoint
CMD ["./run_processing.sh"]