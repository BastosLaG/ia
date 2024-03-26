import datasets
import transformers
import torch

"""
Ici on charge le jeu de données sst2 (jeu d'entraînement et de validation)
sst2 (Stanford sentiment treebank) est un jeu de donnée constitué d'extraits de commentaires de films
et de leur polarité associée (si ce commentaire est plutot positif, ou plutot négatif)
Plus de détails sur load_dataset au lien suivant: "https://huggingface.co/docs/datasets/loading"
Ces datasets sont des objets Dataset Huggingface sont tout deux composés de trois colonnes:
    - Colonne "sentence": Contient le contenu textuel du commentaire, en format string
    - Colonne "label": Contient la polarité du commentaire (0 pour négatif, 1 pour positif)
    - Colonne "idx": Contient l'index du commentaire dans le jeu de données (pas nécessaire pour nous)
"""

train_dataset, validation_dataset = datasets.load_dataset('glue', 'sst2', split=['train', 'validation'])

# On passe à la tokenization du texte du jeu de données. On va pour ce faire charger un tokenizer déjà disponible dans
# HuggingFace. On peut pour ce faire instancier un objet de la classe AutoTokenizer à partir de son constructeur
# .from_pretrained, auquel on donne le nom du tokenizer désiré. Il existe à peu près autant de tokenizer que de
# modèles sur huggingface.

tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')


# On définit la fonction de preprocessing qu'on va appliquer à nos deux datasets
def preprocess(dataset):
    """

    :param dataset: Un objet Dataset HuggingFace contenant notre jeu de données. Doit posséder au moins une colonne
    "sentence" contenant du texte à tokenizer
    :return: Un objet Dataset HuggingFace contenant toutes les colonnes du Dataset d'entrée, plus les sorties du
    tokenizer. Contient en particulier une colonne 'input_ids' avec les versions tokenizée des phrases d'entrée
    """

    # On commence par définir la transformation que l'on veut appliquer à un bloc de ligne d'un dataset donné. Cette
    # Fonction prend en entrée un dictionnaire, dont les clés correspondent aux colonnes de nos datasets, et dont les
    # valeurs correspondent à une liste de valeurs de chaque colonne.

    def tokenize_batch(batch):
        """

        :param batch: Un dictionnaire dont les clés correspondent aux colonne de dataset, et dont les valeurs
        correspondantes sont des listes d'éléments des colonnes correspondantes
        :return: Un dictionnaire avec de nouvelles clés correpondant aux nouvelles colonnes du dataset, et dont les
        valeurs correspondantes sont des lite de transformées des valeurs d'entrées
        """
        # Dans cette fonction, on cherche simplement à tokenizer les phrases présentes dans la liste
        # batch['sentence']. On va également les tronquer à 128 tokens, pour éviter d'avoir des séquences d'entrées
        # trop longues
        pre_out = tokenizer(
            batch['sentence'],  # la liste de string que l'on veut tokenizer
            max_length=128,  # la longueur maximale que l'on accepte pour une séquence de tokens,
            truncation='longest_first'  # La façon dont le tokenizer doit couper les séquences trop longues
        )

        return pre_out

    # On applique la fonction encode_batch de manière distribuée à notre dataset en appelant la méthode .map
    new_dataset = dataset.map(
        tokenize_batch,  # Le Callable que l'on souhaite appliqué à notre dataset
        batched=True,  # Indique si l'on veut exécuter le Callable sur des blocs de lignes du dataset ou ligne par ligne
    )

    # On renomme la colonne du Dataset contenant notre variable à prédire "labels" pour éviter les confusions avec le
    # trainer huggingface
    new_dataset = new_dataset.rename_column("label", "labels")
    new_dataset = new_dataset.remove_columns([x for x in new_dataset.column_names if x not in ['labels', 'input_ids']])

    return new_dataset


# On applique notre fonction de preprocessing à nos deux datasets
tokenized_train_dataset = preprocess(train_dataset)
tokenized_validation_dataset = preprocess(validation_dataset)


# On implémente notre modèle en créant une classe héritant de torch.nn.Module
class MyModel(torch.nn.Module):
    def __init__(self, num_classes, num_tokens, hidden_size, num_layers, padding_idx=0):
        """
        Dans le __init__ d'un module, on vient instancier tous le paramètres du modèle, ou le cas échéant, toutes les
        couches de neurones qu'on utilisera dans le modèle (qui sont également des objet héritant de torch.nn.Module)

        :param num_classes: Nombre d'état de la variable à prédire (1 pour une variable quantitative)
        :param num_tokens: Nombre de vecteurs de mots à initialiser pour la couche d'embedding (correspond à la taille
        du vocabulaire du tokenizer)
        :param hidden_size: Nombre de neurones par couches (on décide arbitrairement que toutes les couches du modèle
        auront le même nombre de neurones, ca marche plutôt bien pour les données textuelles)
        :param num_layers: Nombre de couches de neurones à convolution qu'on va utiliser
        :param padding_idx: Indice correpondant au padding dans les tenseurs d'entrée
        """

        # On appelle la méthode __init__ de la classe parent
        super(MyModel, self).__init__()

        # On définit notre couche d'embedding, qui convertira nos tokens en vecteurs compatibles avec notre modèle
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=num_tokens,  # Définit le nombre de vecteurs de mots de la couche d'embedding
            embedding_dim=hidden_size,  # Définit la taille des vecteur de mots
            padding_idx=padding_idx,  # Définit quel token sera considéré comme le token de padding
        )

        # On définit les couches de notre réseau à convolution. Pour ce faire, on instancie un objet
        # torch.nn.ModuleList, que l'on instancie avec une liste de nos couches. Utiliser ModuleList est nécessaire,
        # afin que pytorch puisse accéder aux paramètres de chaque couche (notamment pour les ajuster pendant la
        # descente de gradient).
        self.conv_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    in_channels=hidden_size,  # Nombre de cartes d'activations en entrée, fixé à hidden_size
                    out_channels=hidden_size,  # Nombre de neurones de la couche (et donc nombre de cartes d'activations en sortie), fixé à hidden_size
                    kernel_size=10,  # Taille des sous-matrice auxquelles auront accès les neurones de la couche
                    padding='same',  # Padding pour assurer que les séquences en sortie de la couche aient la même longueur que celles en entrée
                ) for _ in range(num_layers)  # On répète l'instanciation num_layers fois pour avoir num_layers couches
            ]
        )

        # On définit la couche finale de prédiction linéaire du modèle, comme une instance de torch.nn.Linear
        self.prediction_layer = torch.nn.Linear(
            in_features=hidden_size,  # Le nombre de variables explicatives pour la couche de prédiction
            out_features=num_classes, # Le nombre d'états de la variable à prédire (1 pour une variable quantitative)
        )

    def forward(self, inputs):
        """
        On implémente la logique de la couche, c'est à dire la transformation âramétrique que l'on veut appliquer aux
        variables d'entrée (ici le texte du commentaire) pour obtenir une prédiction de la variable cible (ici la
        polarité du commentaire)

        :param inputs: Un tenseur de dimension (num_observations, taille_de_séquence) contenant les tokens des phrases
        des commentaires. Ce tenseur a un dtype d'entier (typiquement dtype Long)
        :return: Les prédictions du modèle, comme un tenseur torch de dimension (num_observations, self.num_classes)
        """

        # On appelle la couche self.embedding_layer sur les inputs pour obtenir nos séquences de vecteurs de mots
        embedding_outputs = self.embedding_layer(inputs)

        # embedding_outputs est un tenseur de dimension (num_observations, taille_de_séquence, self.hidden_size).
        # Or, les couches torch.nn.Conv1d s'attendent à recevoir un tenseur de dimension
        # (num_observation, self.hidden_size, taille_de_séquence). Il faut donc permuter les deux dimensions 1 et 2
        # de notre tenseur afin de le rendre compatible avec nos couches de convolution, ce que l'on peut faire avec
        # la méthode .transpose d'un tenseur
        outputs = embedding_outputs.transpose(1, 2)

        # On fait passer nos séquences de vecteurs de mots dans notre réseau à convolution
        for conv_layer in self.conv_layers:
            # On fait passer nos variables cachées dans une couche à convolution
            outputs = conv_layer(outputs)

            # On fait passer nos variables cachées dans une fonction d'activation (pour rendre le modèle non-linéaire)
            outputs = torch.nn.GELU()(outputs)

        # On réduit nos séquences de vecteurs à un unique vecteur de dimension hidden_size en additionnant entre eux
        # les vecteurs de chaque séquence
        outputs = outputs.sum(
            axis=-1  # axis=-1 car on veut des tenseurs de forme (num_obs, hidden_size)
        )

        # On applique notre couche de prédiction linéaire sur cette dernière sortie pour obtenir nos prédictions
        outputs = self.prediction_layer(outputs)

        return outputs


# On instancie notre modèle
model = MyModel(
    num_classes=2,  # On cherche à prédire une variable catégorielle à deux états (commentaire négatif ou positif)
    num_tokens=tokenizer.vocab_size,  # On donne à notre modèle la taille du vocabulaire de notre tokenizer
    hidden_size=128,  # On décide (un peu arbitrairement) du nombre de neurones par couche de notre modèle
    num_layers=8,  # On décide (un peu arbitrairement) du nombre de couches à convolution de notre modèle
    padding_idx=tokenizer.pad_token_id  # On donne à notre modèle le token de padding utilisé par notre tokenizer
)


# Maintenant que notre jeu de données et notre modèle sont prêt, il ne nous reste plus qu'à ajuster ce dernier! On
# va pour ce faire implémenter un Trainer adapté à nos besoins
class CustomTrainer(transformers.Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Calcule la fonction objectif de model à partir de inputs, dans le but d'exécuter une itération de descente
        de gradient

        :param model: Une instance du modèle que l'on cherche à ajuster
        :param inputs: Un dictionnaire avec au moins deux clés "input_ids" et "labels" pointant respectivement vers un
        tenseur de variables d'entrée (les commentaires tokenizés) et un tenseur de variable de sortie (0 ou 1 suivant
        la polarité du commentaire)
        :param return_outputs: Un booléen influant sur le comportement de la méthode.

        :return: La valeur de la fonction objectif de model évalué sur inputs si return_outputs=False, ou un tuple
        (valeur de la fonction objectif, prédictions) si return_outputs=True
        """

        # On sort nos variables à prédire du dictionnaire d'entrée, et on les reshape de manière à être compatible avec
        # l'objet qui calcule notre fonction objectif
        labels = inputs.get('labels').view(-1)

        # On sort les variables d'entrée du dictionnaire d'entrée
        input_ids = inputs.get("input_ids")

        # On appelle notre modèle comme un Callable sur nos variables d'entrées pour obtenir ses prédictions, en format
        # de log_probabilité (puisqu'on n'utilise pas le SoftMax pour des raisons de stabilité numérique)
        log_predictions = model(input_ids)

        # On calcule la fonction objectif, comme l'entropie croisée entre les log predictions et les variables à prédire
        # On commence par instancier un objet CrossEntropyLoss
        loss_function = torch.nn.CrossEntropyLoss()

        # Puis on l'évalue sur nos prédictions et nos labels pour obtenir la loss. C'est cette valeur qui sera utilisée
        # pour effectuer une itération de descente de gradient
        loss = loss_function(log_predictions, labels)

        return (loss, log_predictions) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Calcule les prédictions de model à partir de inputs, par exemple dans le but d'évaluer les performances du
        modèle. Cette fonction n'est pas utilisée lors d'une itération de descente de gradient, et peut donc s'implémenter
        sous un statement with torch.no_grad(): pour limiter la consommation mémoire du programme.

        :param model: Une instance du modèle que l'on cherche à évaluer
        :param inputs: Un dictionnaire avec au moins deux clés "input_ids" et "labels" pointant respectivement vers un
        tenseur de variables d'entrée (les commentaires tokenizés) et un tenseur de variable de sortie (0 ou 1 suivant
        la polarité du commentaire)
        :param prediction_loss_only: Un booléen influant sur le comportement de la méthode
        :param ignore_keys: Pas utilisé dans notre cas (et globalement j'ai jamais trop compris à quoi ca servait)

        :return: La valeur de la fonction objectif de model évalué sur inputs si return_outputs=False, ou un tuple
        (valeur de la fonction objectif, prédictions) si return_outputs=True
        """

        # On implémente notre fonction sous un statement 'with torch.no_grad()' pour limiter la consommation mémoire
        # du programme
        with torch.no_grad():
            # On sort nos variables à prédire du dictionnaire d'entrée, et on les reshape de manière à être compatible avec
            # l'objet qui calcule notre fonction objectif
            labels = inputs.get('labels').view(-1)

            # On sort les variables d'entrée du dictionnaire d'entrée
            input_ids = inputs.get("input_ids")

            # On appelle notre modèle comme un Callable sur nos variables d'entrées pour obtenir ses prédictions, en format
            # de log_probabilité (puisqu'on n'utilise pas le SoftMax pour des raisons de stabilité numérique)
            log_predictions = model(input_ids)

            # On calcule la fonction objectif, comme l'entropie croisée entre les log predictions et les variables à prédire
            # On commence par instancier un objet CrossEntropyLoss
            loss_function = torch.nn.CrossEntropyLoss()

            # Puis on l'évalue sur nos prédictions et nos labels pour obtenir la loss. Ici cette valeur ne sera pas
            # utilisée pour effectuer une itération de descente de gradient, mais ca reste une métrique intéressante
            # à évaluer, notamment pour identifier des cas de surrajustement
            loss = loss_function(log_predictions, labels)

        return (loss, log_predictions, labels)


# On va maintenant définir la fonction compute_metric qui permettra au trainer d'évaluer les performances du modèle!
def compute_metric_fn(eval_predictions):
    """
    Calcule les métriques d'évaluation du modèle.

    :param eval_predictions: Un objet huggingface EvalPrediction, constitué de deux attributs 'predictions', et 'label_ids'
    correspondant respectivement aux prédictions du modèle et aux véritables valeurs des variables à prédire
    :return: Un dictionnaire python, contenant des couples clés/valeurs avec le nom de la métrique, puis sa valeur
    """

    # On sort les predictions de l'objet EvalPrediction d'entrée
    predictions = eval_predictions.predictions

    # On sort les valeurs des variables explicatives de l'objet EvalPrediction d'entrée
    labels = eval_predictions.label_ids

    # On choisit la prédiction du modèle comme l'état de la variable à prédire associé à la plus haute probabilité
    predictions = predictions.argmax(-1)

    # On calcule la quantité moyenne de prédictions correctes
    accuracy = (predictions == labels).mean()

    # On instancie notre dictionnaire de métriques (qui contient içi une métrique)
    metric_dictionary = {"accuracy": accuracy}

    return metric_dictionary


# Il ne nous reste plus qu'à instancier un objet CustomTrainer pour ajuster notre modèle! On commence par instancier
# un objet TrainingArguments qui gère toute la logistique de l'ajustement

training_args = transformers.TrainingArguments(
    output_dir="output_dir",  # Le répertoire dans lequel le Trainer écrira tout un tas d'infos sur l'ajustement
    learning_rate=0.0001,  # Le pas d'apprentissage de la descente de gradient, choisi un peu arbitrairement
    num_train_epochs=5,  # Le nombre d'epoch que durera l'ajustement du modèle
    per_device_train_batch_size=64,  # Le nombre d'observation qu'on tirera au sort pour effectuer chaque itération de la descente de gradient stochastique
    per_device_eval_batch_size=1024, # Permet de faire l'évaluation des performances sur le jeu de test en plusieurs étapes pour éviter les problèmes de mémoire
    evaluation_strategy='steps',  # Indique au Trainer si vous voulez évaluer les performances du modèle toutes les epochs, ou après un certain nombre d'itération de descente
    eval_steps=200,  # Si vous avez choisi 'steps' en evaluation_strategy, indique la fréquence à laquelle vous voulez évaluer les performances du modèle
    remove_unused_columns=False,  # Important de le mettre en False, sinon ca marche pas. J'ai jamais compris à quoi ca servait
    save_strategy='no',  # Indique si vous voulez sauvegarder votre modèle dans 'output_dir'. Je mets 'no' parceque ici je veux pas enregistrer de modèles
)

# On peut maintenant instancier notre CustomTrainer!
trainer = CustomTrainer(
    model=model,  # Indique au trainer quel modèle on cherche à ajuster
    args=training_args,  # Indique au trainer tous les comportements qu'on a spécifié dans training_args
    train_dataset=tokenized_train_dataset,  # Indique au trainer le jeu de données à utiliser pour ajuster model
    eval_dataset=tokenized_validation_dataset,  # Indique au trainer le jeu de données à utiliser pour évaluer model
    compute_metrics=compute_metric_fn,
    data_collator=transformers.DataCollatorWithPadding(tokenizer)  # Indique au modèle qu'il va avoir besoin de mettre du padding sur les séquences d'entrée pour en faire des Tensor
)

# Et finalement on ajuste notre modèle en appelant la méthode .train de notre objet Trainer
trainer.train()