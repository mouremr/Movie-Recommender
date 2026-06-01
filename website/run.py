
from website.app import create_app
import gunicorn

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5001)