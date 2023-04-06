import subprocess
import datetime
import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort,render_template
from flask_sqlalchemy import SQLAlchemy
import os
import pickle

model = pickle.load(open('knn.pkl','rb'))



# customize your API through the following parameters
classes_path = './data/labels/coco.names'
weights_path = './weights/yolov3.tf'
tiny = False                    # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
output_path = './detections/'   # path to output folder where images with detections are saved
num_classes = 80                # number of classes in model

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')

# Initialize Flask application
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String(80),unique=True,nullable=False)
    password=db.Column(db.String(80),nullable=False)
    def __repr__(self):
        return '<User %r>' % self.username

with app.app_context():
 db.create_all()





# @app.route('/',methods=['GET','POST'])
# def first():
#     return render_template('first.html')

# @app.route('/login',methods=['GET','POST'])
# def login():
#     return render_template('login.html')

# @app.route('/register',methods=['GET','POST'])
# def register():
    # if request.method=='POST':
    #     username=request.form.get('username')
    #     password=request.form.get('password')
    #     user = User(username=username,password=password)
#         db.session.add(user)
#         db.session.commit()
#         return ("Done")


#     return render_template('register.html')

@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/gallery',methods=['GET','POST'])
def gallery():
    return render_template('gallery.html')

@app.route('/testimonial',methods=['GET','POST'])
def testimonial():
    return render_template('testimonial.html')

@app.route('/faq',methods=['GET','POST'])
def faq():
    return render_template('faq.html')

@app.route('/about',methods=['GET','POST'])
def about():
    return render_template('about.html')

@app.route('/dimensions',methods=['GET','POST'])
def dimensions():
    return render_template('dashboard.html')

@app.route('/truck',methods=['GET','POST'])
def det_truck():
    data = request.get_json()    
    bed = int(data[0][1])
    refrigerator =int(data[4][1]) 
    sofa = int(data[2][1])
    bench = int(data[3][1])
    diningtable= int(data[1][1])

    arr = np.array([bed,diningtable,sofa,bench,refrigerator])
    arr = arr.reshape(1,5)


    perdictions = model.predict(arr)[0]


    return jsonify({'result':perdictions})



@app.route('/part1',methods=['GET','POST'])
def part1():
    data=request.get_json()
    global pa
    pa=data[0][1]
    global da 
    da=data[1][1]

    return jsonify({'status':'success'})

@app.route('/part2',methods=['GET','POST'])
def part2():
    data = request.get_json()
    global fragile
    fragile=data[0][1]
    global truck
    truck=data[1][1]
    global result
    result=data[2][1]
    global distance
    distance=data[3][1]
    global date 
    date=data[4][1]

    return jsonify({'status':'success'})

@app.route('/bill',methods=['GET','POST'])
def bill():
    
    pa1 = pa
    da1 = da
    distance1 = distance
    truck1 = truck
    fragile1 = fragile
    result1 = result
    date1 = date
    bdate = datetime.datetime.now()
    pdate = bdate.strftime("%y-*m-*d")

    return render_template('bill.html',pa=pa1,da=da1,pdate=pdate,date=date1,distance=distance1,fragile=fragile1,truck=truck1,result=result1)

# API that returns JSON with classes found in images
@app.route('/detections', methods=['POST'])
def get_detections():
    raw_images = []
    images = request.files.getlist("images")
    image_names = []
    for image in images:
        image_name = image.filename
        image_names.append(image_name)
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        raw_images.append(img_raw)
        
    num = 0
    
    # create list for final response
    response = []

    for j in range(len(raw_images)):
        # create list of responses for current image
        responses = []
        raw_img = raw_images[j]
        num+=1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100))
            })
        response.append({
            "image": image_names[j],
            "detections": responses
        })
        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
        print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))

    #remove temporary images
    for name in image_names:
        os.remove(name)
    try:
        return jsonify({"response":response}), 200
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)