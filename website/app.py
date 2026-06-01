from flask import *
import os
from website.reccommender import *
from werkzeug.utils import secure_filename
import pandas as pd

# UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # upload_dir = os.path.join(base_dir, 'uploads')
    
    app = Flask(__name__)
    # app.config['UPLOAD_FOLDER'] = upload_dir
    
    @app.route("/")
    def home():
       return render_template("Index.html")

    @app.route('/recommend', methods=['POST'])
    def recommend():
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                user_data = process_file(filepath)
        elif 'username' in request.form and request.form["username"] != "":
            username = request.form["username"]
            user_data = process_username(username)
        
        results = run_recommender(user_data)

        return render_template('results.html', table=results.to_html(classes='data'))
    
    return app