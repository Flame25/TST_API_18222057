from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime
import pickle
import onnx
from dotenv import load_dotenv
from supabase import create_client
import os
import jwt
from datetime import datetime, timedelta
from better_profanity import profanity
import random

# Load the ONNX model and tokenizer
model_path = "model.onnx"

with open("tokenizer.pickle", "rb") as handle: 
    tokenizer = pickle.load(handle)

try:
    # Initialize ONNX Runtime and tokenizer
    session = onnxruntime.InferenceSession(model_path)
    label_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
except Exception as e:
    raise RuntimeError(f"Error initializing model or tokenizer: {str(e)}")

model = onnx.load("model.onnx")
# Get the model's graph
graph = model.graph

# Print the input details
for input_tensor in graph.input:
    print(f"Name: {input_tensor.name}")
    print(f"Type: {input_tensor.type.tensor_type.elem_type}")
    dims = input_tensor.type.tensor_type.shape.dim
    shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in dims]
    print(f"Shape: {shape}")


# Initialize Flask app
app = Flask(__name__)

CORS(app, origins=["*"])

def verify_jwt(token):
    try:
        decoded = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def generate_jwt(api_key):
    payload = {
        "sub": api_key,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=30),
    }
    return jwt.encode(payload, os.getenv("SECRET_KEY"), algorithm="HS256")

def check_api_key(api_key): 
    # Load Data from .env
    load_dotenv()
    # Get the Supabase URL and API Key from environment variables
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # Initialize the Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Check API Key
    response = supabase.table("user_data").select("*").eq("public_key",api_key).execute()

    return response.data

def texts_to_sequences_with_random(tokenizer, texts):
    # Get the size of the vocabulary
    vocab_size = len(tokenizer.word_index) + 1  # Include 1 to account for padding

    sequences = []
    for text in texts:
        sequence = []
        for word in text.split():
            word_id = tokenizer.word_index.get(word)
            if word_id:  # If the word exists in the tokenizer
                sequence.append(word_id)
            else:  # Assign a random token ID for unknown words
                random_id = random.randint(1, vocab_size - 1)  # Exclude padding (ID=0)
                sequence.append(random_id)
        sequences.append(sequence)
    return sequences

@app.route("/auth/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        if "user_name" not in data:
            return jsonify({"status": "Register Failed", "error": "Username is required"}),200
        if "password" not in data:
            return jsonify({"status": "Register Failed", "error": "Password is required"}),200

        # Check pass and user name in database

        load_dotenv()

        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        response = supabase.table("user_data").select("*").eq("user_name", data["user_name"]).execute()

        if response.data:
            return jsonify({"status": "Register Failed", "error": "Username found. Please Login!"}),200

        data_reg = {
            "user_name": data["user_name"],
            "password": data["password"],
            "status": "Basic"
        }

        response_register = supabase.table("user_data").insert(data_reg).execute()

        # Check the response
        if response_register.data:
            return jsonify({"status": "Register Success"})
        else:
            return jsonify({"status": "Register Failed", "error": "Try again later"}),200

    except Exception as e:
        return jsonify({"status": "Failed", "error": f"{str(e)}"}),500


@app.route("/auth/login", methods=["POST"])
def login():
    try:
        data = request.get_json()

        if "user_name" not in data:
            return jsonify({"status": "Login Failed", "error": "Username is required"}),200
        if "password" not in data:
            return jsonify({"status": "Login Failed", "error": "Password is required"}),200

        # Check pass and user name in database

        load_dotenv()

        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        response = supabase.table("user_data").select("*").eq("user_name", data["user_name"]).execute()

        if not response.data:
            return jsonify({"status": "Login Failed", "error": "Username not found. Please register first!"}),200

        if response.data[0]["password"] != data["password"]:
            return jsonify({"status": "Login Failed", "error": "Wrong password"}),200

        # TODO: Add last login session in database
        return jsonify({"status": "Login Success", "token": generate_jwt(response.data[0]["public_key"])}),200
    except Exception as e:
        return jsonify({"status": "Error", "error": f"{str(e)}"}),400

@app.route("/api/predict", methods=["POST"])
def predict_sentiment():
    try:
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"status": "Auth Failed","error": "Authorization header missing or invalid"}), 401

        decoded_token = verify_jwt(token)
        if not decoded_token:
            return jsonify({"status": "Auth Failed","error": "Invalid or expired token"}), 401

        # Check API Key on Token
        if(not check_api_key(decoded_token["sub"])):
            return jsonify({"status": "Invalid Key", "error": "API Key is invalid"}), 402

        # Parse input JSON
        data = request.get_json()

        if "text" not in data:
            return jsonify({"status": "Failed","error": "Input text is required"}), 400
        # Tokenize the input text
        inputs = texts_to_sequences_with_random(tokenizer, data["text"])
        
        # Prepare inputs for ONNX Runtime
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: inputs})

        # Extract predictions
        logits = outputs[0]
        prediction = logits.argmax(axis=-1)[0]
        sentiment = label_mapping[prediction]

        # Return result
        return jsonify({"text": data["text"], "sentiment": sentiment})
    except Exception as e:
        return jsonify({"status": "Failed", "error": f"Inference error: {str(e)}"}), 500

@app.route("/api/filter", methods=["POST"])
def filter_text():
    try:
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"status": "Auth Failed","error": "Authorization header missing or invalid"}), 401

        decoded_token = verify_jwt(token)
        if not decoded_token:
            return jsonify({"status": "Auth Failed","error": "Invalid or expired token"}), 401
        if(not check_api_key(decoded_token["sub"])):
            return jsonify({"status": "Invalid Key", "error": "API Key is invalid"}),402

        data = request.get_json()

        if "text" not in data:
            return jsonify({"status": "Failed", "error": "Input Text is required"}), 400

        # Program Logic

        return jsonify({"text": data["text"],"response":profanity.censor(data["text"])})
    except Exception as e:
        return jsonify({"status": "Failed", "error": f"{str(e)}"}),500

@app.route("/api/bot", methods=["POST"])
def filter_and_check():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "Auth Failed","error": "Authorization header missing or invalid"}), 401

        token = auth_header.split(" ")[1]
        decoded_token = verify_jwt(token)
        if not decoded_token:
            return jsonify({"status":"Auth Failed","error": "Invalid or expired token"}), 401
        if(not check_api_key(decoded_token["sub"])):
            return jsonify({"status":"Invalid Key", "error": "API Key is invalid"}),402

        # TODO:Make a checker if the key is premium only
    except Exception as e:
        return jsonify({"status" : "Failed", "error": f"{str(e)}"}),500

@app.route("/auth/token",methods=["POST"])
def check_key():
    try:
        data = request.get_json()
        if "api_key" not in data: 
            return jsonify({"status": "Failed", "error": "API Key is required"}),402

        if not check_api_key(data["api_key"]):
            return jsonify({"status": "Invalid Key", "error": "API Key not valid"}),402

        return jsonify({"status": "Success", "token": generate_jwt(data["api_key"])})
    except Exception as e:
        return jsonify({"status": "Failed", "error": f"{str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8057)
