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

1. `__init__()`: This is the constructor method which is called when an instance of the class is initialised. It sets up the logger, loads data, and prints out initial information about the model and data.
    
    ```python
    def __init__(self, model_name, data_directory, model_directory, use_amp=False):
        self.model_directory = model_directory
        self.data_directory = data_directory
        self.model_name = model_name
        ...
        print(f"Objet crée avec succes voici le chemin du fichier de log de Tensor Board : {os.path.join(self.model_directory,'log_tensor_board')}")
    
    ```
    
2. `load_data()`: This function loads the data from a specified directory, concatenates all the data frames, and stores them as a single data frame.
    
    ```python
    def load_data(self):
        self.logger.info("Chargement des données...")
        self.logger.info(f"     Liste des fichiers trouvés dans le dossier de data au chemin specifié:")
        ...
        print(f"Données chargées avec succès en {end_time - start_time} secondes.\\n")
    
    ```
    
3. `preprocess_text()`: This function preprocesses textual data by converting the text into lowercase and storing it in a new column.
    
    ```python
    def preprocess_text(self, column_names):
        ...
        self.logger.info(f"Texte pré-traité avec succès en {end_time - start_time} secondes.")
        print(f"Texte pré-traité avec succès en {end_time - start_time} secondes.")
    
    ```
    
4. `split_data()`: This function splits the data into training, validation and testing sets, and then tokenizes and masks the data. It also creates DataLoaders for each set.
    
    ```python
    def split_data(self, X="original_text_lower", y="reference_summary_lower", train_size=0.6, val_size=0.2, batch_size=32):
        ...
        self.logger.info(f"Données divisées et dataloaders instanciés avec succès en {time.time() - start_time} secondes. \\n")
    
    ```
    
5. `train()`: This function trains the model using the specified parameters. It also implements early stopping and gradient clipping.
    
    ```python
    def train(self, epochs=4, learning_rate=0.001, weight_decay=0.01, grad_clip=None, early_stopping_patience=None, log_interval=10, validation_interval=100):
        ...
        self.logger.info(f"Modèle entraîné et enregistrer avec succès en {time.time() - start_time} \\n")
    
    ```
    
6. `evaluate()`: This function evaluates the model on the validation set during training.
    
    ```python
    def evaluate(self):
        ...
        print(f"La perte moyenne sur l'ensemble de validation est {avg_loss}")
        return avg_loss
    
    ```
    
7. `summarize()`: This function generates a summary for a given text.
    
    ```python
    def summarize(self, text):
        ...
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    ```
    
8. `final_evaluate()`: This function evaluates the model on the test set after training, and calculates evaluation metrics like Rouge scores, cosine similarity, and BLEU score.
    
    ```python
    def final_evaluate(self):
        ...
        print("Évaluation finale du modèle terminée avec succès.")
    
    ```
    
9. `__str__()`: This function returns a string representation of the class instance, providing information about the model, data, and evaluation results.
    
    ```python
    def __str__(self):
        ...
        return f"Résumeur : ..."
    
    ```
