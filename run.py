from app import create_app
import app.services.datastore as datastore
import pandas as pd

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
