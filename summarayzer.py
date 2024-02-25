import os, json, time, torch, uuid, logging
from datetime import datetime
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    BertForSequenceClassification,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
    T5Tokenizer,
    BertTokenizer,
    AutoConfig,
)
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence


class Summarayzer(object):
    def __init__(self, model_name, data_directory, class_name):

        if "bert" in class_name:
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name, model_max_length=512
            )
            self.model = BertForSequenceClassification.from_pretrained(model_name)
        elif "t5" in class_name:
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name, model_max_length=512
            )
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            raise ValueError("Erreur - Entrez un nom correct (BERT - T5)")

        # Vérifiez si un GPU est disponible et si nous sommes sur un OS compatible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda" and os.name != "nt"
        self.class_name = class_name

        self.model = self.model.to(self.device)

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.model_directory = os.getcwd()
        while os.path.exists(self.model_directory):
            unique_id = str(uuid.uuid4()).replace("-", "")  # ID unique sans tirets
            self.model_directory = os.path.join(os.getcwd(), f"{unique_id}")[
                :80
            ]  # Nom unique

        os.makedirs(self.model_directory, exist_ok=True)

        self.writer = SummaryWriter(
            os.path.join(self.model_directory, "log_tensor_board")
        )
        self.data_directory = data_directory
        self.model_name = model_name
        self.model_lowloss = None
        self.data = None
        self.evaluation_results = {
            "low_loss": {
                "rouge_scores": None,
                "similarity_with_reference": None,
                "similarity_with_original": None,
                "bleu_score": None,
            },
            "most_trained": {
                "rouge_scores": None,
                "similarity_with_reference": None,
                "similarity_with_original": None,
                "bleu_score": None,
            },
        }

        # Initialisation du logger
        self.logger = logging.getLogger("LOGGER")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(
            os.path.join(self.model_directory, "details.log")
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        self.logger.info(f"Classe du modele : {self.model_name}")
        self.logger.info(f"Nom du modele : {self.model.__class__.__name__}")
        self.logger.info(
            f"Log Tensor Board : {os.path.join(self.model_directory,'log_tensor_board')}"
        )

        self.load_data()

        print(
            f"Objet crée avec succes voici le chemin du fichier de log de Tensor Board : {os.path.join(self.model_directory,'log_tensor_board')}"
        )

    def load_data(self):
        self.logger.info("Chargement des données...")
        self.logger.info(
            f"     Liste des fichiers trouvés dans le dossier de data au chemin specifié:"
        )
        print("Chargement des données...")
        start_time = time.time()
        dfs = []
        common_columns = None
        for file_name in os.listdir(self.data_directory):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.data_directory, file_name)
                self.logger.info(f"         {file_name}")
                with open(file_path, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame(data.values())
                if common_columns is None:
                    common_columns = set(df.columns)
                else:
                    common_columns = common_columns.intersection(df.columns)
                dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        self.data = df[list(common_columns)]
        end_time = time.time()

        self.logger.info(
            f"     Votre modèle et ses résultats seront stockés dans le dossier {self.model_directory} dans l'espace courant"
        )
        self.logger.info(
            f"     Données chargées avec succès en {end_time - start_time} secondes."
        )
        self.logger.info(
            f"     Forme des données chargées dans le data frame  :  {self.data.shape}.\n"
        )

        print(
            f"Votre modèle et ses résultats seront stockés dans le dossier {self.model_directory} dans l'espace courant"
        )
        print(f"Données chargées avec succès en {end_time - start_time} secondes.\n")

    def preprocess_text(self, column_names):
        self.logger.info("Pré-traitement du texte...")
        print("Pré-traitement du texte...")
        start_time = time.time()
        for column_name in column_names:
            tokenised_column = self.data[
                column_name
            ].str.lower()  # Convertir en minuscules
            self.data[column_name + "_lower"] = tokenised_column
        end_time = time.time()
        self.logger.info(
            f"Texte pré-traité avec succès en {end_time - start_time} secondes."
        )
        print(f"Texte pré-traité avec succès en {end_time - start_time} secondes.")

    def split_data(
        self,
        X="original_text_lower",
        y="reference_summary_lower",
        train_size=0.6,
        val_size=0.2,
        batch_size=32,
    ):
        self.logger.info("Division des données...")
        print("Division des données...")
        start_time = time.time()
        X_data = self.data[X].tolist()
        y_data = self.data[y].tolist()
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_data, y_data, test_size=1 - train_size, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size / (1 - train_size), random_state=42
        )

        self.logger.info(
            f"Pourcentage de répartition dans train : {len(X_train)/len(X_data)*100}%, val : {len(X_val)/len(X_data)*100}%, test : {len(X_test)/len(X_data)*100}%"
        )

        def tokenize_and_mask(texts):
            tokenized = [
                self.tokenizer.encode_plus(
                    text, truncation=True, padding="max_length", max_length=512
                )
                for text in texts
            ]
            input_ids = pad_sequence(
                [torch.tensor(t["input_ids"]) for t in tokenized], batch_first=True
            )
            attention_masks = pad_sequence(
                [torch.tensor(t["attention_mask"]) for t in tokenized], batch_first=True
            )
            return input_ids, attention_masks

        X_train, train_masks = tokenize_and_mask(X_train)
        X_val, val_masks = tokenize_and_mask(X_val)

        def tokenize(texts):
            tokenized = [
                self.tokenizer.encode_plus(
                    text, truncation=True, padding="max_length", max_length=512
                )
                for text in texts
            ]
            input_ids = pad_sequence(
                [torch.tensor(t["input_ids"]) for t in tokenized], batch_first=True
            )
            return input_ids

        y_train = tokenize(y_train)
        y_val = tokenize(y_val)

        data_train = [
            [x, x_mask, y] for x, x_mask, y in zip(X_train, train_masks, y_train)
        ]
        data_val = [[x, x_mask, y] for x, x_mask, y in zip(X_val, val_masks, y_val)]
        data_test = [[X, y] for X, y in zip(X_test, y_test)]

        num_workers = 4 if os.name == "posix" else 0

        # Créer les DataLoaders
        train_dataloader = DataLoader(
            data_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_dataloader = DataLoader(
            data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_dataloader = DataLoader(
            data_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.data = {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
        }
        self.logger.info(f"Taille du batch choisis : {batch_size}")
        self.logger.info(
            f"Taille des dataloaders - train : {len(train_dataloader)}, val : {len(val_dataloader)}, test : {len(test_dataloader)}"
        )
        self.logger.info(
            f"Données divisées et dataloaders instanciés avec succès en {time.time() - start_time} secondes. \n"
        )

        print(
            f"Données divisées et dataloaders instanciés avec succès en {time.time() - start_time} secondes."
        )

    def train(
        self,
        epochs=4,
        learning_rate=0.001,
        weight_decay=0.01,
        grad_clip=None,
        early_stopping_patience=None,
        log_interval=10,
        validation_interval=100,
    ):
        print("Debut de l'entrainement ... ")
        self.logger.info("Debut de l'entrainement ... ")
        self.logger.info(
            f"     Parametres d'entrainement  =>  epochs = {epochs}, learning_rate = {learning_rate}, weight_decay = {weight_decay}, grad_clip = {grad_clip}, early_stopping_patience = {early_stopping_patience}, log_interval = {log_interval}, validation_interval = {validation_interval} "
        )

        start_time = time.time()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        total_steps = len(self.data["train"]) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        best_loss = float("inf")
        patience_counter = 0
        step = 0

        try:
            for epoch in range(epochs):
                self.model.train()
                for batch in self.data["train"]:
                    inputs = batch[0].to(self.device)
                    masks = batch[1].to(self.device)
                    labels = batch[2].to(self.device)

                    outputs = self.model(
                        input_ids=inputs, attention_mask=masks, labels=labels
                    )
                    loss = outputs.loss

                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip
                        )
                    scheduler.step()
                    optimizer.zero_grad()

                    if step % log_interval == 0:
                        print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
                        self.writer.add_scalar(
                            "Training Loss",
                            loss.item(),
                            epoch * len(self.data["train"]) + step,
                        )

                    if step % validation_interval == 0:
                        val_loss = self.evaluate()
                        self.writer.add_scalar(
                            "Validation Loss",
                            val_loss,
                            epoch * len(self.data["val"]) + step,
                        )
                        self.logger.info(
                            f"     Validation Loss : {val_loss}, Epoch : {epoch} , Batch : {step}"
                        )

                        if val_loss < best_loss:
                            best_loss = val_loss
                            os.makedirs(
                                os.path.join(self.model_directory, "low_loss"),
                                exist_ok=True,
                            )
                            torch.save(
                                self.model.state_dict(),
                                os.path.join(
                                    self.model_directory,
                                    "low_loss",
                                    "pytorch_model.bin",
                                ),
                            )
                            config = self.model.config
                            config.save_pretrained(
                                os.path.join(self.model_directory, "low_loss")
                            )
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if (
                                early_stopping_patience
                                and patience_counter >= early_stopping_patience
                            ):
                                print("Early stopping triggered.")
                                self.logger.info("Early stopping triggered.")
                                return

                    step += 1
        except Exception as e:
            print(f"Error occurred during training: {e}")
            self.logger.info(f"Error occurred during training: {e}")
            raise e

        os.makedirs(os.path.join(self.model_directory, "most_train"), exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_directory, "most_train", "pytorch_model.bin"),
        )
        config = self.model.config
        config.save_pretrained(os.path.join(self.model_directory, "most_train"))
        print(
            f"Modèle entraîné et enregistrer avec succès en {time.time() - start_time} ."
        )
        self.logger.info(
            f"Modèle entraîné et enregistrer avec succès en {time.time() - start_time} \n"
        )

    def evaluate(self):
        """
        Évalue le modèle sur le jeu de validation pendant l'entraînement.
        Utilise self.val_dataloader comme DataLoader pour l'évaluation.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(self.data["val"]):
                inputs = batch[0].to(self.device)
                masks = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                # Passage en avant
                outputs = self.model(
                    input_ids=inputs, attention_mask=masks, labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()

                # Enregistrez la perte pour chaque lot
                self.writer.add_scalar("Loss/evaluate", loss.item(), i)

        avg_loss = total_loss / len(self.data["val"])

        print(f"La perte moyenne sur l'ensemble de validation est {avg_loss}")
        return avg_loss

    def summarize(self, text):
        self.model.eval()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).to(self.device)
        if "bert" in self.class_name:
            outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif "t5" in self.class_name:
            outputs = self.model.generate(**inputs, max_length=50)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def final_evaluate(self):
        """
        Évalue le modèle sur le jeu de test après l'entraînement.
        Utilise self.test_dataloader comme DataLoader pour l'évaluation.
        """
        print(f"Debut de l'evaluation finale ...")
        self.logger.info(f"Debut de l'evaluation finale ...")
        self.logger.info(f"     Resultats d'evaluation finale : ...")

        # Charger le modèle avec la perte la plus faible pendant l'entraînement
        self.model_lowloss = self.model.__class__.from_pretrained(
            os.path.join(self.model_directory, "low_loss")
        )

        # Créer un dictionnaire pour stocker les résultats des deux modèles
        self.evaluation_results = {"most_trained": {}, "low_loss": {}}

        # Évaluer les deux modèles
        for model_name, model in [
            ("most_trained", self.model),
            ("low_loss", self.model_lowloss),
        ]:
            model.eval()
            reference_summaries = []
            automatic_summaries = []
            original_texts = []

            with torch.no_grad():
                for batch in self.data["test"]:
                    texts_ori = batch[0]
                    texts_summ = batch[1]

                    for text, sum in zip(texts_ori, texts_summ):
                        encoded_input = self.tokenizer.encode_plus(
                            text,
                            max_length=512,
                            truncation=True,
                            padding="max_length",
                        )

                        input_ids = (
                            torch.tensor(encoded_input["input_ids"])
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        attention_mask = (
                            torch.tensor(encoded_input["attention_mask"])
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        reference_summaries.append(sum)
                        original_texts.append(text)

                        if "bert" in self.class_name:
                            output = model.generate(
                                input_ids,
                                max_new_tokens=50,
                                attention_mask=attention_mask,
                            )
                            automatic_summaries.append(
                                self.tokenizer.decode(
                                    output.prediction, skip_special_tokens=True
                                )
                            )

                        elif "t5" in self.class_name:
                            output = model.generate(
                                input_ids,
                                max_new_tokens=50,
                                attention_mask=attention_mask,
                            )

                            automatic_summaries.append(
                                self.tokenizer.decode(
                                    output[0], skip_special_tokens=True
                                )
                            )
                        print(f"liste des résumés automatiques : {automatic_summaries}")
                        print(f"liste des résumés reference : {reference_summaries}")

            # Calcul des scores d'évaluation
            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            scores = [
                scorer.score(ref, hyp)
                for ref, hyp in zip(reference_summaries, automatic_summaries)
            ]
            rouge_scores = {
                key: np.mean([score[key].fmeasure for score in scores])
                for key in scores[0].keys()
            }

            model = SentenceTransformer("all-MiniLM-L6-v2")
            automatic_embeddings = model.encode(automatic_summaries)
            reference_embeddings = model.encode(reference_summaries)
            original_embeddings = model.encode(original_texts)
            similarity_with_reference = cosine_similarity(
                automatic_embeddings, reference_embeddings
            )
            similarity_with_original = cosine_similarity(
                automatic_embeddings, original_embeddings
            )

            bleu_score = corpus_bleu(
                reference_summaries,
                automatic_summaries,
                smoothing_function=SmoothingFunction().method1,
            )

            self.evaluation_results[model_name] = {
                "rouge_scores": rouge_scores,
                "similarity_with_reference": np.mean(similarity_with_reference),
                "similarity_with_original": np.mean(similarity_with_original),
                "bleu_score": bleu_score,
            }

            self.logger.info(
                f"         Modele {model_name} : {self.evaluation_results[model_name]}"
            )

        print("Évaluation finale du modèle terminée avec succès.")

    def __str__(self):
        try:
            data_status = "Pas encore chargé"
            data_size = "N/A"
            if self.data is not None:
                if isinstance(self.data, pd.DataFrame):
                    data_status = "Chargé mais pas encore divisé"
                    data_size = str(self.data.shape)
                elif isinstance(self.data, dict):
                    data_status = "Dataloaders prêts"
                    data_size = ", ".join(
                        [f"{k}: {len(v)}" for k, v in self.data.items()]
                    )

            return f"""
            Résumeur :
            - Répertoire du modèle : {self.model_directory}
            - Nom du modèle : {self.model_name}
            - Classe du modèle : {self.model.__class__.__name__}
            - Répertoire des données : {self.data_directory}
            - État des données : {data_status}
            - Taille des données : {data_size}
            - writer : {'Instancié avec le type ' + str(type(self.writer))}
            - Résultats d'évaluation (plus entrainé) : {'Calculé avec les valeurs ' + str(self.evaluation_results['most_trained']) if self.evaluation_results['most_trained']['rouge_scores'] is not None else "Pas encore calculé"}
            - Résultats d'évaluation (plus petite perte) : {'Calculé avec les valeurs ' + str(self.evaluation_results['low_loss']) if self.evaluation_results['low_loss']['rouge_scores'] is not None else "Pas encore calculé"}
            """
        except Exception as e:
            return f"Erreur lors de la création de la chaîne de caractères : {e}"
