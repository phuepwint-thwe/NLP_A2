# A2 - LSTM Text Generator
- Student Information
- Overview
- Features
- Installation
- Dataset
- Web Application Features
- Credit

## Student Information
- Name : Phue Pwint Thwe
- ID : st124784

## Overview
This project demonstrates the implementation of an LSTM-based text generator. The model was trained on the *War and Peace* text dataset to generate coherent and contextually relevant text. A web application was developed using Streamlit to allow users to interact with the model.

## Features
1. **Train an LSTM Model:**
   - Processes text data to create a language model.
   - Learns to generate text based on a given prompt.
2. **Interactive Web Application:**
   - Users can input prompts, adjust creativity level (temperature), and specify the length of generated text.
   - Outputs the generated text in real-time.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository_link>
   cd <repository_directory>
   ```

2. **Install Dependencies:**
   Make sure you have Python 3.7 or above installed. Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset used is the text of *War and Peace*. You can download it from a public source or provide your own dataset in `.txt` format.
- Place the dataset file in the project directory.
- Update the file path in the code if necessary.

## Running the Project

### Train the Model
The Jupyter notebook provided contains the code for training the LSTM model. To train:
1. Open the notebook (`A2_LSTM_Text_Generator.ipynb`).
2. Run all cells to preprocess the dataset and train the model.
3. Save the trained model.

### Run the Web Application
To launch the Streamlit web application:
1. Ensure the trained model is saved in the appropriate directory.
2. Run the following command in the terminal:
   ```bash
   streamlit run app.py
   ```
3. Open `localhost:8501` in your web browser to access the application.

## Web Application Features
1. **Prompt Input:**
   - Enter a text prompt (e.g., "What a delightful").
2. **Adjustable Parameters:**
   - Select creativity level (temperature).
   - Choose the length of the generated text.
3. **Real-time Text Generation:**
   - View the generated continuation of your prompt.

## File Structure
- `A2_LSTM_Text_Generator.ipynb`: Notebook for training the LSTM model.
- `app.py`: Code for the web application.
- `requirements.txt`: List of dependencies.
- `war_and_peace.txt`: Sample dataset (if provided).

## Example Outputs
### Prompt: "What a delightful"
- **Output 1:** "What a delightful expenditure product defeats arch crafty guiding accurately."
- **Output 2:** "What a delightful planets interesting enjoying shout antique weep roman lagging pause."

## Future Improvements
1. Enhance model coherence through additional fine-tuning.
2. Add support for multiple datasets.
3. Include metrics for evaluating text quality.

## Credits
- Dataset: *War and Peace* by Leo Tolstoy.
- Frameworks: PyTorch, Streamlit, and HuggingFace's `datasets` library.

---