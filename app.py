from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
from src.pdf_utils import pdf_to_text
from src.vector_store import upsert_documents
from src.vector_store import query_vector_store
from src.llm_client import generate_response


app = Flask(__name__)
app.secret_key = "super-secret-key"  # Required for flash messages

# Folder to store uploaded files
UPLOAD_FOLDER = "resources"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf", "txt"}

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Unified Default Route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Combined UI for chat and upload

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    """Handle file upload, extraction, and upsert process."""
    if request.method == "POST":
        # Check if a file is included in the request
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            try:
                # Save the uploaded file
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)

                # Extract text and upsert to vector store
                extracted_text = (
                    pdf_to_text(file_path)
                    if filename.endswith(".pdf")
                    else open(file_path, "r", encoding="utf-8").read()
                )
                documents = [{"id": filename, "content": extracted_text}]
                upsert_documents(documents)

                # Return JSON success response
                return jsonify({
                    "status": "success",
                    "message": f"File '{filename}' uploaded and processed successfully."
                }), 200

            except Exception as e:
                # Return JSON error response for exceptions
                return jsonify({
                    "status": "error",
                    "message": f"Error processing file: {str(e)}"
                }), 500
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid file type. Only PDF and TXT files are allowed."
            }), 400

    # Default return for GET requests
    return render_template("upload.html")


    

@app.route("/chat", methods=["GET"])
def chat():
    """Render the chat UI."""
    return render_template("chat.html")

@app.route("/query", methods=["POST"])
def query():
    """Handle user input, retrieve vector store context, and return a response."""
    try:
        # Parse the incoming JSON data
        data = request.get_json()
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"response": "Please enter a valid message."}), 400

        # Step 1: Retrieve context from the vector store
        context = query_vector_store(user_input)  # Fetch relevant content
        # print("Retrieved context:", context)  # Debugging

        # Step 2: Generate a response using the OpenAI API
        response_text = generate_response(user_input, context)
        # print("Generated response:", response_text)  # Debugging

        # Step 3: Return the response as a JSON string
        return jsonify({"response": response_text}), 200


    except Exception as e:
        # Handle any exceptions and return an error response
        error_message = f"Error: {str(e)}"
        print(error_message)  # Debugging
        return jsonify({"response": error_message}), 500


if __name__ == "__main__":
    app.run(debug=True)
