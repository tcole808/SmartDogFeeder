from flask import Flask, redirect, url_for
from login_home import auth_bp
from home import home_bp
from animal_ml import animal_bp

app = Flask(__name__)
app.secret_key = "dev"

app.register_blueprint(auth_bp)
app.register_blueprint(home_bp)
app.register_blueprint(animal_bp)

@app.route("/")
def index():
    return redirect(url_for("auth.login"))

if __name__ == "__main__":
    app.run(debug=True)