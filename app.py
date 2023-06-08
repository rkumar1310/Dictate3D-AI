import pickle
import re

import joblib
import spacy
from flask import Flask, jsonify, request
from keras.models import load_model
from nltk import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, BertTokenizer
from word2number import w2n

app = Flask(__name__)


# Load the saved model
color_model = load_model("./color-detection-model/color_model.h5")

# Load the tokenizer
with open("./color-detection-model/tokenizer.pickle", "rb") as handle:
    color_tokenizer = pickle.load(handle)
# Load the models
ner_model = spacy.load("./command-trained-model")
# Load the models and tokenizer from the specified folder
tokenizer = BertTokenizer.from_pretrained("./models/tokenizer")
model_general = BertForSequenceClassification.from_pretrained("./models/model_general")
model_specific = BertForSequenceClassification.from_pretrained(
    "./models/model_specific"
)
# Use LabelEncoder for the labels
le_general = joblib.load("./labels/le_general.pkl")
le_specific = joblib.load("./labels/le_specific.pkl")
ACTIONS_MAP = {
    "add": [
        "add_cone",
        # "add_copy",
        "add_cube",
        "add_cylinder",
        "add_light",
        "add_plane",
        "add_pyramid",
        "add_sphere",
        "add_torus",
    ],
    "change_material": [
        "change_material_brick",
        "change_material_cloth",
        "change_material_concrete",
        "change_material_copper",
        "change_material_diamond",
        "change_material_dull",
        "change_material_glass",
        "change_material_glossy",
        "change_material_gold",
        "change_material_granite",
        "change_material_iron",
        "change_material_metal",
        "change_material_plastic",
        "change_material_random",
        "change_material_rubber",
        "change_material_sandstone",
        "change_material_shiny",
        "change_material_silver",
        "change_material_steel",
        "change_material_stone",
        "change_material_velvet",
        "change_material_wood",
    ],
    "change_opacity": ["change_opacity"],
    "color": [
        "color_change",
    ],
    # "flip": ["flip_horizontal", "flip_vertical"],
    "move": [
        "move_away",
        "move_backward",
        "move_closer",
        "move_down",
        "move_forward",
        "move_left",
        "move_right",
        "move_up",
    ],
    "remove": ["remove", "remove_light"],
    "rotate": ["rotate_anticlockwise", "rotate_clockwise"],
    "scale": [
        "scale_down",
        # "scale_match",
        "scale_up",
    ],
    "start_animation": ["jump", "spin"],
    "stop_animation": ["jump", "spin"],
    "tilt": ["tilt"],
}


@app.route("/process", methods=["POST"])
def encode():
    data = request.get_json()
    command = data.get("command")
    results = predict_command(command)
    first_result = results[0]  # Access the first element of the results array

    general_label = first_result[0]  # Access the general label
    specific_label = first_result[1]  # Access the specific label
    total_score = first_result[2]  # Access the total score

    doc = ner_model(command)  # replace with your text
    value = None
    color = None
    has_color = False
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
        if ent.label_ == "Value":
            value = ent.text
        if ent.label_ == "Color":
            has_color = True

    if has_color:
        color = predict_color(command).tolist()
    print(color)

    if value != None:
        value = convert_words_to_numbers(value)

    return jsonify(
        command=command,
        general_label=general_label,
        specific_label=specific_label,
        value=value,
        color=color,
    )


# Function for making predictions
def predict_command(input_sentence, k=10):
    encoding = tokenizer(
        input_sentence, truncation=True, padding=True, return_tensors="pt"
    )

    # Get predictions
    general_output = model_general(
        encoding["input_ids"], attention_mask=encoding["attention_mask"]
    )[0]
    specific_output = model_specific(
        encoding["input_ids"], attention_mask=encoding["attention_mask"]
    )[0]

    # Get top k values and indices
    general_topk_values, general_topk_indices = general_output.topk(k)
    specific_topk_values, specific_topk_indices = specific_output.topk(k)

    # Move tensors back to cpu for numpy operations
    general_topk_values = general_topk_values.detach().cpu().numpy().flatten()
    general_topk_indices = general_topk_indices.detach().cpu().numpy().flatten()
    specific_topk_values = specific_topk_values.detach().cpu().numpy().flatten()
    specific_topk_indices = specific_topk_indices.detach().cpu().numpy().flatten()

    # Use the inverse_transform method to get the original labels
    general_commands = le_general.inverse_transform(general_topk_indices)
    specific_commands = le_specific.inverse_transform(specific_topk_indices)

    # Combine commands with their scores
    general_predictions = [
        (command, float(score))
        for command, score in zip(general_commands, general_topk_values)
    ]
    specific_predictions = [
        (command, float(score))
        for command, score in zip(specific_commands, specific_topk_values)
    ]

    results = []

    # Iterate over all the general predictions
    for general_action, general_score in general_predictions:
        # If this general action is in the action map
        if general_action in ACTIONS_MAP:
            # Get the specific actions for this general action
            specific_actions = ACTIONS_MAP[general_action]
            # For each specific action
            for specific_action, specific_score in specific_predictions:
                # If this specific action is among the specific actions for this general action
                if specific_action in specific_actions:
                    # Compute total score
                    total_score = general_score + specific_score
                    # Append result
                    results.append((general_action, specific_action, total_score))

    # Sort the results by total score in descending order
    results.sort(key=lambda x: x[2], reverse=True)
    return results


# function to preprocess sentence
def preprocess(sentence):
    stop_words = stopwords.words("english")
    return [
        word
        for word in word_tokenize(sentence.lower())
        if word not in stop_words and word.isalpha()
    ]


def predict_color(color_text):
    tokenized_color_text = color_tokenizer.texts_to_sequences([color_text])
    padded_color_text = pad_sequences(tokenized_color_text, maxlen=5)
    predicted_color_rgb = color_model.predict(padded_color_text)
    predicted_color_rgb = (predicted_color_rgb * 255).astype(int)
    return predicted_color_rgb


def convert_words_to_numbers(text):
    # Split on space and dash
    parts = re.split("-| ", text)

    # Try to convert each part to a number
    numbers = []
    for part in parts:
        try:
            # First, try to convert it as a number
            numbers.append(int(part))
        except ValueError:
            # If that doesn't work, try to convert it as a word
            try:
                numbers.append(w2n.word_to_num(part))
            except ValueError:
                print(f"Could not convert '{part}' to a number.")

    # Sum all numbers
    return sum(numbers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
