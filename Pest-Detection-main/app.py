from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import google.generativeai as genai
import os
from werkzeug.utils import secure_filename
import re

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCUzxfdoBC6cjoDoY9Z8Nl6Ju-5f3zCa0A"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Load class labels (use your full list here)
class_labels = [
    "rice leaf roller",
    "rice leaf caterpillar",
    "paddy stem maggot",
    "asiatic rice borer",
    "yellow rice borer",
    "rice gall midge",
    "Rice Stemfly",
    "brown plant hopper",
    "white backed plant hopper",
    "small brown plant hopper",
    "rice water weevil",
    "rice leafhopper",
    "grain spreader thrips",
    "rice shell pest",
    "grub",
    "mole cricket",
    "wireworm",
    "white margined moth",
    "black cutworm",
    "large cutworm",
    "yellow cutworm",
    "red spider",
    "corn borer",
    "army worm",
    "aphids",
    "Potosiabre vitarsis",
    "peach borer",
    "english grain aphid",
    "green bug",
    "bird cherry-oataphid",
    "wheat blossom midge",
    "penthaleus major",
    "longlegged spider mite",
    "wheat phloeothrips",
    "wheat sawfly",
    "cerodonta denticornis",
    "beet fly",
    "flea beetle",
    "cabbage army worm",
    "beet army worm",
    "Beet spot flies",
    "meadow moth",
    "beet weevil",
    "sericaorient alismots chulsky",
    "alfalfa weevil",
    "flax budworm",
    "alfalfa plant bug",
    "tarnished plant bug",
    "Locustoidea",
    "lytta polita",
    "legume blister beetle",
    "blister beetle",
    "therioaphis maculata Buckton",
    "odontothrips loti",
    "Thrips",
    "alfalfa seed chalcid",
    "Pieris canidia",
    "Apolygus lucorum",
    "Limacodidae",
    "Viteus vitifoliae",
    "Colomerus vitis",
    "Brevipoalpus lewisi McGregor",
    "oides decempunctata",
    "Polyphagotars onemus latus",
    "Pseudococcus comstocki Kuwana",
    "parathrene regalis",
    "Ampelophaga",
    "Lycorma delicatula",
    "Xylotrechus",
    "Cicadella viridis",
    "Miridae",
    "Trialeurodes vaporariorum",
    "Erythroneura apicalis",
    "Papilio xuthus",
    "Panonchus citri McGregor",
    "Phyllocoptes oleiverus ashmead",
    "Icerya purchasi Maskell",
    "Unaspis yanonensis",
    "Ceroplastes rubens",
    "Chrysomphalus aonidum",
    "Parlatoria zizyphus Lucus",
    "Nipaecoccus vastalor",
    "Aleurocanthus spiniferus",
    "Tetradacus c Bactrocera minax",
    "Dacus dorsalis(Hendel)",
    "Bactrocera tsuneonis",
    "Prodenia litura",
    "Adristyrannus",
    "Phyllocnistis citrella Stainton",
    "Toxoptera citricidus",
    "Toxoptera aurantii",
    "Aphis citricola Vander Goot",
    "Scirtothrips dorsalis Hood",
    "Dasineura sp",
    "Lawana imitata Melichar",
    "Salurnis marginella Guerr",
    "Deporaus marginatus Pascoe",
    "Chlumetia transversa",
    "Mango flat beak leafhopper",
    "Rhytidodera bowrinii white",
    "Sternochetus frigidus",
    "Cicadellidae"
]

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 102  # Update if your number of classes is different
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model_path = r"C:\Users\Jefferson\Desktop\PROJECTS\Pest_Detection\resnet50_0.497.pkl"  # Update path if needed
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# PyTorch image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ai_pest_description(pest_name):
    try:
        prompt = f"Provide a brief description of {pest_name}, including its impact on crops. Limit to 40-50 words."
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating pest description: {e}")
        return f"Description for {pest_name} is currently unavailable."

def get_ai_management_advice(pest_name):
    try:
        prompt = f"Provide 5 specific management strategies for controlling {pest_name} in crops. Each point should be 30-40 words. Include both organic and conventional methods. Format as bullet points."
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating management advice: {e}")
        return "Unable to generate management advice at this time. Please try again later."

def generate_ai_response(user_message, pest_context):
    try:
        system_prompt = "You are an agricultural pest management assistant helping farmers identify and manage crop pests."
        if pest_context:
            system_prompt += f" Given the detected pest: [pest_context], provide a brief description of the pest (40–50 words) . Format the strategies as bullet points, and ensure the advice is actionable and relevant to field conditions"
        full_prompt = f"{system_prompt}\n\nUser: {user_message}"
        response = gemini_model.generate_content(full_prompt)
        cleaned_response = re.sub(r'\s*\*+\s*', ' ', response.text).strip()
        return cleaned_response
    except Exception as e:
        print(f"Error generating chat response: {e}")
        return "I'm sorry, I couldn't process your request at this time. Please try again later."

def detect_pests(image_path):
    # Read and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        class_index = torch.argmax(probabilities, dim=1).item()
        confidence = float(probabilities[0, class_index].item()) * 100

    # Get the predicted pest
    pest_name = class_labels[class_index]

    # Get pest description and AI-generated management advice
    description = get_ai_pest_description(pest_name)
    management_advice = get_ai_management_advice(pest_name)

    # Return the prediction result with AI-enhanced information
    return {
        'detected': confidence > 50,  # Threshold can be adjusted
        'pest': pest_name,
        'confidence': confidence,
        'description': description,
        'management_advice': management_advice
    }

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Process the uploaded image
        result = detect_pests(file_path)
        return jsonify(result), 200
    return jsonify({'message': 'Invalid file type'}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    user_message = data['message']
    pest_context = data.get('pest', '')
    # Generate AI response using the chat context
    response = generate_ai_response(user_message, pest_context)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
