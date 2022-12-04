import pickle

from flask import Flask, request, jsonify, url_for


def predict_single(customer, dv, model):
    X = dv.transform(customer)
    y_pred = model.predict_proba(X)[:1]
    return y_pred[0]


def load(file_name):
    with open(file_name, 'rb') as f_in:
        return pickle.load(f_in)


model = load('model1.bin')
dv = load('dv.bin')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    print('predict enter')
    print(request)
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)[1]
    card = prediction >= 0.5

    print(prediction)
    result = {
        'card_propobility': float(prediction),
        'card': bool(card),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,  port=9696)

with app.test_request_context():
    print(url_for('predict'))

