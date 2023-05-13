import os

# Flask
from flask import Flask, request, render_template, Response, jsonify
# from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from keras.models import load_model
from keras.preprocessing import image

model_path = 'Best weight/model/model.h5'
model = load_model(model_path, compile=False)
print('Model loaded. Start serving...')

# Declare a flask app
app = Flask(__name__)


# print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(path, model):
    """
    :param path:
    :return:
    """
    img = image.load_img(path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = x / 255.0
    x = x.reshape(1, 128, 128, 3)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x, mode='tf')

    prediction = model.predict(x)
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
    """
    :return:
    """
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
                return render_template('index.html', result="PNEUMONIA", src=img_path)  # jsonify(result="PNEUMONIA")
            else:
                return render_template('index.html', result="NORMAL")  # jsonify(result="NORMAL")
    except:
        return render_template('index.html', result="Please upload an image")


if __name__ == '__main__':
    app.run(port=5002, threaded=False, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
