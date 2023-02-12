from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


model = pickle.load(open("D:/hackathon/nit/crop_prediction.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    features = request.form.to_dict()

    
    prediction = model.predict([list(features.values())])

    
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run()





