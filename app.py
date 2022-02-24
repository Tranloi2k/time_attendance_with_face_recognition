
from flask import Flask, render_template, flash, jsonify, redirect, url_for
from flask_socketio import SocketIO
from flask_socketio import send, emit
import socket
import cv2
from train import train
from deepface import DeepFace
import tensorflow as tf
import src.facenet as facenet
import numpy as np
import pickle
from flask import request
from geopy.geocoders import Nominatim
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import io
from PIL import Image
import base64
import re
from pymongo import MongoClient
from pandas import DataFrame
from random import randint
import time
from create_newclass import add_data




#from mtcnn import MTCNN
#detector = MTCNN()

client = MongoClient('localhost', 27017)
#Getting the database instance
db = client['test']
coll = db['acount']
coll2 = db['history']

sender_address = 'tranloi162710@gmail.com'
sender_pass = 'vsmoqnvrhsyxxsea'
receiver_address = 'loitv@dyno.vn'

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
metrics = ["cosine", "euclidean", "euclidean_l2"]
FACENET_MODEL_PATH = '20180402-114759.pb'
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = "my_classifier.pkl"
best_name='Vui lòng chờ '
temp = ''
login = True
check = False
admin = False
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

facenet.load_model(FACENET_MODEL_PATH)

# Get input and output tensors
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

app = Flask(__name__)
app.secret_key = "super secret key"
socketio = SocketIO(app)

def predict(frame):
    cropped = DeepFace.detectFace(img_path=frame,
                                  detector_backend=backends[0], enforce_detection=False)
    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                        interpolation=cv2.INTER_CUBIC)
    scaled = facenet.prewhiten(scaled)
    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
    emb_array = sess.run(embeddings, feed_dict=feed_dict)

    predictions = model.predict_proba(emb_array)
    return predictions
img=cv2.imread(r"C:\Users\Admin\PycharmProjects\facenet\data\my_data\bich_phuong\bich-phuong.jpg")
predict(img)

def create_id():
    x= False
    while x == False:
        new_id = random_with_N_digits(6)
        myquery = {'id': str(new_id)}
        mydoc = coll.find(myquery).limit(1)
        mydoc = list(mydoc)
        if len(mydoc)==0:
            x = True
            return new_id

def create_id2():
    x= False
    while x == False:
        new_id = random_with_N_digits(6)
        myquery = {'id': str(new_id)}
        mydoc = coll2.find(myquery).limit(1)
        mydoc = list(mydoc)
        if len(mydoc)==0:
            x = True
            return new_id

def random_with_N_digits(n):
    range_start = 10 ** (n - 1)
    range_end = (10 ** n) - 1
    return randint(range_start, range_end)

def send_mail(acc, name ,time, location):
    mail_content = 'Xin chào ' +str(name)+'''
Bạn đã chấm công vào lúc: ''' +str(time)+' tại: ' + str(location)
    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = 'receiver_address'
    message['Subject'] = 'Chấm công online.'  # The subject line
    # The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(sender_address, sender_pass)  # login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, acc, text)
    session.quit()
    print('Mail Sent')

@app.route('/')
def index():

    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home_2.html')

@socketio.on('name')
def handle_message():
    global best_name
    global check
    emit('name', {'data': best_name, 'check': check})

@socketio.on('name2')
def handle_message2():
    global process

    emit('name2', {'process': process})

@app.route('/login', methods=["GET","POST"])
def login():

    if request.method =='POST':
        try:
            global acc
            acc = request.form['acc']
            password = request.form['pass']
            print(str(acc))
            print(str(password))
            myquery = {'acc': str(acc)}
            mydoc = coll.find(myquery).limit(1)
            mydoc = list(mydoc)
            temp = DataFrame(mydoc)
            global login
            if len(temp) != 0:
                if str(password) == str(temp['password'].values[0]):
                    global id
                    global admin
                    id = temp['id'].values[0]
                    role = temp['admin'].values[0]
                    if role == 'True':
                        admin = True
                        login = True
                        return redirect(url_for('submit'))

                    else:
                        print(id)
                        login = True
                        return render_template('home_2.html')
                else:
                    flash("Mật khẩu không đúng ", "success")
                    return redirect(url_for('login'))
            else:
                flash("Tài khoản không tồn tại ", "success")
                return redirect(url_for('login'))

        except:
            global newacc
            global newpassword

            newacc = request.form['newacc']
            print(str(newacc))
            newpassword = request.form['newpassword']
            newpassword1 = request.form['newpassword1']
            myquery = {'acc': str(newacc)}
            mydoc = coll.find(myquery).limit(1)
            mydoc = list(mydoc)
            temp = DataFrame(mydoc)

            if newpassword == newpassword1:
                if len(temp)==0:
                    if newacc:
                        global ma
                        ma = random_with_N_digits(6)
                        mail_content = str(ma)
                        # Setup the MIME
                        message = MIMEMultipart()
                        message['From'] = sender_address
                        message['To'] = str(newacc)
                        message['Subject'] = 'Mã xác nhận'  # The subject line
                        # The body and the attachments for the mail
                        message.attach(MIMEText(mail_content, 'plain'))
                        # Create SMTP session for sending the mail
                        session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
                        session.starttls()  # enable security
                        session.login(sender_address, sender_pass)  # login with mail_id and password
                        text = message.as_string()
                        session.sendmail(sender_address, str(newacc), text)
                        session.quit()
                        global start_time
                        start_time = time.time()
                        return redirect(url_for('verify'))
                else:
                    flash("email đã được đăng ký ", "success")
                    return redirect(url_for('login'))
            else:
                flash("mật khẩu không khớp ", "success")
                return redirect(url_for('login'))


    else:
        return render_template('login.html')

@app.route('/new_password', methods=['GET','POST'])
def new_password():
    if request.method == 'POST':
        global acc1, newpass
        acc1 = request.form['acc']
        newpass = request.form['pass']
        newpass1 = request.form['pass1']
        if newpass == newpass1:
            global ma2
            ma2 = random_with_N_digits(6)
            mail_content = str(ma2)
            # Setup the MIME
            message = MIMEMultipart()
            message['From'] = sender_address
            message['To'] = str(acc1)
            message['Subject'] = 'Mã xác nhận'  # The subject line
            # The body and the attachments for the mail
            message.attach(MIMEText(mail_content, 'plain'))
            # Create SMTP session for sending the mail
            session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
            session.starttls()  # enable security
            session.login(sender_address, sender_pass)  # login with mail_id and password
            text = message.as_string()
            session.sendmail(sender_address, str(acc1), text)
            session.quit()
            global start_time
            start_time = time.time()
            return redirect(url_for('verify2'))
        else:

            return redirect(url_for('new_password'))
    else:
        return render_template('new_password.html')


@app.route('/verify', methods =['GET','POST'])
def verify():
    if request.method == 'POST':
        output = request.form['ma']
        end_time = time.time()
        if (start_time-end_time)<=60:
            if str(ma) == str(output):
                new_acc = {"id": str(create_id()), "acc": str(newacc), "password": str(newpassword), "admin": "False"}
                coll.insert_one(new_acc)
                flash("Tạo tài khoản thành công", "success")
                return redirect(url_for('login'))
            else:
                flash("Mã xác nhận không đúng", "success")
                return redirect(url_for('login'))
        else:
            flash("Mã đã hết hạn ", "success")
            return redirect(url_for('login'))
    else:
        return render_template('verify.html')

@app.route('/verify2', methods =['GET','POST'])
def verify2():
    if request.method == 'POST':
        output = request.form['ma']
        end_time = time.time()
        if (start_time-end_time)<=60:
            if str(ma2) == str(output):
                filter = {'acc': str(acc1)}
                newvalues = {"$set": {'password': str(newpass)}}
                coll.update_one(filter, newvalues)
                flash("Tạo mật khẩu mới thành công", "success")
                return redirect(url_for('login'))
            else:
                flash("Mã xác nhận không đúng", "success")
                return redirect(url_for('new_password'))
        else:
            flash("Mã xác nhận hết hạn ", "success")
            return redirect(url_for('new_password'))
    else:
        return render_template('verify2.html')

@app.route('/logout')
def logout():
    global login
    global admin
    login = False
    admin = False
    return render_template('home.html')

@app.route('/info', methods=['GET','POST'])
def info():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        birthday = request.form['birthday']
        gender = request.form['gender']
        address = request.form['address']
        phone = request.form['phone']
        filter = {'acc': str(acc)}
        newvalues = {"$set": {'first_name': str(first_name),
                              'last_name': str(last_name),
                              'birthday': str(birthday),
                              'gender': str(gender),
                              'address': str(address),
                              'phone': str(phone),
                              }}
        coll.update_one(filter, newvalues)
        return render_template('user_info.html')
    else:

        return render_template('info.html')

@app.route('/test', methods=['POST'])
def test():
    if request.method == 'POST':
        output=request.get_json()
        latitude = output['latitude']
        longitude = output['longitude']
        type = output['type']
        position = str(latitude)+', '+str(longitude)
        # calling the nominatim tool
        geoLoc = Nominatim(user_agent="GetLoc")
        locname = geoLoc.reverse(position)
        current_time = datetime.now()

        print("Current Time =", current_time)
        print(locname.address)

        myquery = {'acc': str(acc)}
        mydoc = coll.find(myquery).limit(1)
        mydoc = list(mydoc)
        temp = DataFrame(mydoc)
        name = temp['first_name'].values + " " + temp['last_name'].values
        name = ' '.join(name)
        new_checkin = {"id":str(create_id2()),"acc": str(acc),"name":str(name), "location": str(locname.address),
                   "time": str(current_time), "type":str(type), "status":"Chờ phê duyệt"}
        coll2.insert_one(new_checkin)
        send_mail(acc, best_name, current_time, locname.address)

        return render_template('index.html')
    else:
        return render_template("index.html")

@app.route('/video', methods=['GET','POST'])
def video():
    if login == True:
        if request.method == 'POST':
            image_b64 = request.values['imageBase64']
            image_data = base64.b64decode(re.sub('^data:image/.+;base64,', '', image_b64))
            image_PIL = Image.open(io.BytesIO(image_data))
            image_np = np.array(image_PIL)
            frame = cv2.cvtColor(np.array(image_np), cv2.COLOR_BGRA2RGB)
            predictions = predict(frame)

            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[
                np.arange(len(best_class_indices)), best_class_indices]

            global best_name
            global id
            global check
            check = False
            prd_id = class_names[best_class_indices[0]]
            print(prd_id)
            print(id)
            if best_class_probabilities < 0.3:
                best_name = 'Vui lòng chờ '
                check = False
            else:
                myquery = {'acc': str(acc)}
                mydoc = coll.find(myquery).limit(1)
                mydoc = list(mydoc)
                temp = DataFrame(mydoc)

                if prd_id == id:
                    check = True
                    best_name = 'Xin chào '+temp['last_name'].values[0]
                else:
                    best_name = 'Vui lòng chờ '
                    check = False

            print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))


            return ''

        else:
            return render_template('index.html')
    else:
        return redirect(url_for('login'))


@app.route('/data', methods = ['GET'])
def data():
    global acc
    myquery = {'acc': str(acc)}
    mydoc = coll.find(myquery).limit(1)
    mydoc = list(mydoc)
    temp = DataFrame(mydoc)
    name = temp['first_name'].values+" "+temp['last_name'].values
    name=' '.join(name)
    gender = temp['gender'].values
    gender = ' '.join(gender)
    birthday = temp['birthday'].values
    birthday = ' '.join(birthday)
    address= temp['address'].values
    address = ' '.join(address)
    phone = temp['phone'].values
    phone = ' '.join(phone)


    return jsonify({'name': str(name),
                    'gender': str(gender),
                    'birthday': str(birthday),
                    'phone': str(phone),
                    'address': str(address),
                    'acc':str(acc)})


@app.route('/new_data', methods=['GET','POST'])
def new_data():
    if login == True:
        if request.method == 'POST':
            #filter = {'acc': 'fan'}
            #newvalues = {"$set": {'quantity': 25}}
            #coll.update_one(filter, newvalues, upsert=True)

            image_b64 = request.values['imageBase64']
            image_data = base64.b64decode(re.sub('^data:image/.+;base64,', '', image_b64))
            image_PIL = Image.open(io.BytesIO(image_data))
            image_np = np.array(image_PIL)
            frame = cv2.cvtColor(np.array(image_np), cv2.COLOR_BGRA2RGB)
            add_data(frame, id)
            train()
            global model, class_names
            with open(CLASSIFIER_PATH, 'rb') as file:
                model, class_names = pickle.load(file)
            print("Custom Classifier, Successfully loaded")
            global process
            process = str('done')

            return render_template('new_data.html')
        else:
            return render_template('new_data.html')
    else:
        return redirect(url_for('login'))

@app.route('/historydata', methods=['GET','POST'])
def historydata():
    global acc
    if request.method == 'GET':
        myquery = {'acc': str(acc)}
        mydoc = coll2.find(myquery)
        mydoc = list(mydoc)
        temp = DataFrame(mydoc)
        temp1 = temp[['location', 'type','time','status']]
        a=[]
        for i in range(0, len(temp1)):
            q = temp1.iloc[i]
            a.append(q.to_dict())
        return {'data': a}

@app.route('/history')
def history():
    if login == True:
        return render_template('history.html')
    else:
        return redirect(url_for('login'))

@app.route('/user_info', methods=['GET', 'POST'])
def user_info():
    if login == True:
        return render_template('user_info.html')
    else:
        return redirect(url_for('login'))
@app.route('/home_3')
def home_3():
    return render_template('home_3.html')

@app.route('/submit')
def submit():
    if admin:
        #myquery = {'acc': str(acc)}
        duyet=coll2.count_documents({'status': str('Duyệt')})
        loai = coll2.count_documents({'status': str('Loại')})
        cho = coll2.count_documents({'status': str('Chờ phê duyệt')})
        mydoc = coll2.find()
        mydoc = list(mydoc)
        temp = DataFrame(mydoc)
        temp1 = temp[['id','name','location', 'type', 'time','status']]
        return render_template("submit.html", column_names=list(temp1.columns),
                               row_data=temp1.values,
                               zip=zip, duyet=duyet,cho=cho,loai=loai)
    else:
        return redirect(url_for('login'))

@app.route('/submit2', methods=['POST'])
def submit2():
    if request.method == 'POST':
        output = request.get_json()
        id = output['id']
        status = output['status']
        filter = {'id': str(id)}
        newvalues = {"$set": {'status': str(status)}}
        coll2.update_one(filter, newvalues)
        return ''

if __name__ == "__main__":
    socketio.run(app,host='localhost', port=8080, debug=True)
