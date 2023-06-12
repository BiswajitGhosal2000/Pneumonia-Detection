import os
import PIL

# Flask
from flask import Flask, request, render_template
from gevent.pywsgi import WSGIServer

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model_path = 'models/resnet152_model.hdf5'
model = load_model(model_path)
print('Model loaded. Start serving...')

app = Flask(__name__)


def model_predict(path, model):
    """
    :param model:
    :param path:
    :return:
    """
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = x.reshape(1, 224, 224, 3)

    prediction = model.predict(x)
    # print("NORMAL = ", prediction[0, 0], "PNEUMONIA =", prediction[0, 1])
    result = prediction[0, 0]
    return result


@app.route('/', methods=['GET'])
def index():
    """

    :return:
    """
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = request.files['file']
        img.save("uploads\image.jpg")
        img_path = os.path.join(os.path.dirname(__file__), 'uploads\image.jpg')
        os.path.isfile(img_path)
        result = model_predict(img_path, model)
        print("Result =", result)
        if result > 0.8:
            return render_template('index.html', result="PNEUMONIA")  # jsonify(result="PNEUMONIA")
        else:
            return render_template('index.html', result="NORMAL")  # jsonify(result="NORMAL")


if __name__ == '__main__':
    app.run(port=5002, threaded=False, debug=True)

    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

"""@app.route('/predict', methods=['GET', 'POST'])

def predict():
    try:
        if request.method == 'POST':

            # Get the image from post request
            img = request.files['file']
            img.save("uploads\image.jpg")

            img_path = os.path.join(os.path.dirname(__file__), 'uploads\image.jpg')

            os.path.isfile(img_path)

            result = model_predict(img_path, model)

            print("Result =", result)

            if result > 0.5:
                return render_template('index.html', result="PNEUMONIA")  # jsonify(result="PNEUMONIA")
            else:
                return render_template('index.html', result="NORMAL")  # jsonify(result="NORMAL")
    except:
        return render_template('index.html', result="Please upload an image")
            
"""