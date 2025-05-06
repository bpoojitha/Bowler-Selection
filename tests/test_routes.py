import json
from app import create_app

app = create_app()
client = app.test_client()

def test_suggest_endpoint():
    response = client.post("/suggest", 
                           data=json.dumps({"phase": "DeathOvers", "team": "MI"}),
                           content_type="application/json")
    assert response.status_code == 200
    data = response.get_json()
    assert "bowler" in data
    assert "reason" in data
