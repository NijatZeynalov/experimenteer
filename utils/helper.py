import logging

# Set up logging
# Function to log the dictionary when the button is clicked
def log_dictionary(dictionary):
    logging.basicConfig(filename='utils/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("**** ITERATION STARTED ****")
    for key, value in dictionary.items():
        logging.info(f"{key}: {value}")
