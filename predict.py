import tensorflow as tf
import numpy as np
import io,os,glob,cv2
import sys,argparse
from pdf2image import convert_from_path, convert_from_bytes
from wand.image import Image
from wand.color import Color
import PyPDF2
import cv2

classL = ['Passport','Drivers License','Birth or Marriage Cert']
#dir_path = os.path.dirname(os.path.realpath(__file__))
#image_path=sys.argv[1]
#filename = dir_path +'/' +image_path
print('Initializing network...')
sess = tf.Session()
saver = tf.train.import_meta_graph('trained_variables.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
#UPDATE if changing amount of classes
y_test_images = np.zeros((1, 3))


def classifyImage(path):
    pre_dict = {}
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        image_size=128
        num_channels=1
        images = []

        if ext.lower() in ['.jpg','.png']:
            temp_data = os.path.join(path,f)
            temp_image = cv2.imread(temp_data,0)

        elif ext.lower() in ['.pdf']:
            temp_image = convertPDF(os.path.join(path,f))
        else:
            continue

        #Flag 0 in imread and convertPDF automationally reads in as greyscale img
        temp_image = cv2.resize(temp_image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        images.append(temp_image.reshape(128,128,1))
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)
        x_batch = images.reshape(1, image_size,image_size,num_channels)

        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result=sess.run(y_pred, feed_dict=feed_dict_testing)

        pre_dict = checkDict(pre_dict,classL[list(result[0]).index(max(result[0]))],f)

    print(pre_dict)
    return pre_dict


def checkDict(pre_dict,classif,filename):
    if classif in pre_dict:
        pre_dict[classif].append(filename)
    else:
        pre_dict[classif] = [filename]
    return pre_dict


def convertPDF(path):
    img_buffer=None

    #Get 1st page of PDF
    pdfFileObj = open(path, 'rb')
    src_pdf = PyPDF2.PdfFileReader(pdfFileObj)
    dst_pdf = PyPDF2.PdfFileWriter()
    dst_pdf.addPage(src_pdf.getPage(0))
    pdf_bytes = io.BytesIO()
    dst_pdf.write(pdf_bytes)
    pdf_bytes.seek(0)

    #Convert PDF to PNG
    img = Image(file = pdf_bytes)
    img.convert("png")
    img.background_color = Color('white')
    img.format = 'tif'
    img.alpha_channel = False

    #Convert PNG to cv2-usable PNG
    img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)

    retval = cv2.imdecode(img_buffer,0)
    return retval


var = input("Please enter a folder path: ")
print("Scanning...")
classifyImage(var)
