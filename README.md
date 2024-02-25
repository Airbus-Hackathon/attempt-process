# Summary Model

This script contains a class `SummaryModel` that is used for text summarization.

## Methods

The class `SummaryModel` has several methods:

- **`__init__`**: This method initializes the model, tokenizer, device, logger, and writer. It also sets up a directory for the model.
- **`load_data`**: This method loads data from a specified directory.
- **`preprocess_text`**: This method pre-processes the text data.
- **`split_data`**: This method splits the data into training, validation, and testing sets.
- **`train`**: This method trains the model. It also saves the best model based on validation loss.
- **`evaluate`**: This method evaluates the model on the validation set during the training phase.
- **`summarize`**: This method is used to generate a summary for a given text.
- **`final_evaluate`**: This method evaluates the model on the test set after the training phase.
- **`__str__`**: This method returns a string representation of the class.

## Usage

1. **Initialization**: Create an instance of the `SummaryModel` class.
2. **Load Data**: Call the `load_data` method to load the data.
3. **Pre-process Text**: Use the `preprocess_text` method to process the data.
4. **Split Data**: Use the `split_data` method to divide the data into training, validation, and testing sets.
5. **Train Model**: Use the `train` method to train the model.
6. **Evaluate Model**: The `evaluate` method can be used to evaluate the model on the validation set during the training phase.
7. **Summarize Text**: Use the `summarize` method to generate a summary of a given text.
8. **Final Evaluation**: After training, use the `final_evaluate` method to evaluate the model on the test set.

## Requirements

- PyTorch
- Transformers
- Pandas
- TensorBoard
- Rouge Scorer
- Sentence Transformers

Please make sure you have these packages installed and updated to the latest versions.
