'''
Copyright 2020 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse


def preprocess_fn(image,size_x,size_y):
    '''
    Image pre-processing.
    Opens image as grayscale then normalizes to range 0:1
    input arg: path of image file
    return: numpy array
    '''
    
    #print("preprocess_fn image.shape:",image.shape)

    image = cv2.resize(image, (size_x,size_y), interpolation=cv2.INTER_LINEAR)
    #print(image.shape)
    image = image/255.0
    return image


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()

def runDPU(id,start,dpu,img,box,rec,acc):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    #print("rec:\n",rec)
    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData)
        #dpu.wait(job_id)
        #print("outputData :",outputData)
        '''store output vectors '''
        temp = []
        ## runSize = 1
        for j in range(runSize):
            #out_q[write_index] = np.argmax(outputData[0][j])
            if np.max(softmax(outputData)) >= acc:
                #print(np.max(softmax(outputData)),np.argmax(outputData[0][j]))
                temp = [rec[count][0],rec[count][1],rec[count][2],rec[count][3],np.max(softmax(outputData)),np.argmax(outputData[0][j])]
                box.append(temp)
        
            write_index += 1
        count = count + runSize

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def SlidingWindow(rec,image,winW,winH):
    #print((winW, winH))
    for (x, y, window) in sliding_window(image, stepSize=2,windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        tempw = image.shape[0]/(winW)
        temph = image.shape[0]/(winH)
        temp = tempw*temph
        if image.shape[0]==temp:
            break
        rec.append(( x , y ,x + winW ,y + winH ))

def draw_face(path, boxes_c,landmarks,num):
    img = cv2.imread(path)
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        cv2.rectangle(img, (corpbbox[0], corpbbox[1]),(corpbbox[2], corpbbox[3]), (0, 255, 0), 1)
        cv2.putText(img, '{:.2f}'.format(score),(corpbbox[0], corpbbox[1] - 2),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if num == 3:
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i]) // 2):
                cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
    return img

def resize_input_image_size(image,set_resize):
    if np.maximum(image.shape[0],image.shape[1]) > set_resize:
        idnex = np.where(np.array(image.shape) > set_resize)
        temp = 1
        while np.maximum(image.shape[0],image.shape[1]) > set_resize:
            if image.shape[0] > image.shape[1]:
                scale = set_resize/image.shape[0]
                size_x = int(np.round(image.shape[0] * scale))
                size_y = int(np.array(image.shape[1] * scale))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (size_y,size_x), interpolation=cv2.INTER_LINEAR)
                temp = temp * scale
            if image.shape[1] > image.shape[0]:
                scale = set_resize/image.shape[1]
                size_x = int(np.round(image.shape[0] * scale))
                size_y = int(np.array(image.shape[1] * scale))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (size_y,size_x), interpolation=cv2.INTER_LINEAR)
                temp = temp * scale
        idnex = idnex[0][0]
        scale = temp
        size_x = int(np.round(image.shape[0]))
        size_y = int(np.array(image.shape[1]))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size_y,size_x), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1
    return image,scale

def py_nms(dets, thresh, mode="Union"):
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def app(image_dir,box_image_dir,threads,model):
    time1 = time.time()
    listimage=os.listdir(box_image_dir)
    runTotal = len(listimage)

    global out_q , box
    out_q = [None] * runTotal
    box = []
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    ''' preprocess images '''
    path = os.path.join(box_image_dir,listimage[0])
    print("path:",path)
    image = cv2.imread(path)
    set_resize = 120 #144
    image,scale = resize_input_image_size(image,set_resize)
    rec = []
    
    if np.maximum(image.shape[0],image.shape[1])<= set_resize:
        SlidingWindow_list = [32,48,60,72,84,96,108,120,132,146,158,170,182,194,206,218,230,242,254,266,278,290]
    else:
        SlidingWindow_list = range(12,image.shape[0],5)

    for j in SlidingWindow_list:
        winW = j
        winH = j
        SlidingWindow(rec,image,winW,winH)
        if winH >= image.shape[0] or winW >= image.shape[1]:
            break
    print("preprocess input image.shape:",image.shape)
    rec = np.array(rec)
    #print("preprocess input rec :",rec)
    all_images = []
    ##np.array(rec).shape[0] = 1376
    for i in range(np.array(rec).shape[0]):
        SlidingWindow_images = image[rec[i][1]:rec[i][3],rec[i][0]:rec[i][2]]
        images = preprocess_fn(SlidingWindow_images,32,32)
        all_images.append(np.array(images))

    all_images = np.array(all_images)
    
    '''run threads '''
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    
    for i in range(threads):
        if (i==threads-1):
            end = len(all_images)
        else:
            end = start+(len(all_images)//threads)
        in_q = all_images[start:end]
        #t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q ,box))
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q ,box,rec,0.9))
        threadAll.append(t1)
        start=end

    
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    
    
    #print("out_q:",out_q)
    nms_box = []
    nms_box_id = py_nms(np.array(box)[:,:5],0.6,mode = 'Union')
    #print("nms_box_id :",nms_box_id)
    for i in range(len(nms_box_id)):
        temp = [int(np.round(box[nms_box_id[i]][0]*(1/scale))),int(np.round(box[nms_box_id[i]][1]*(1/scale))),int(np.round(box[nms_box_id[i]][2]*(1/scale))),int(np.round(box[nms_box_id[i]][3]*(1/scale))),box[nms_box_id[i]][4]]
        nms_box.append(temp)

    nms_boxes = np.array(nms_box)
    all_images = []
    print(" ***************************************************************************************************** ")
    image = cv2.imread(path)
    #print("image.shape:",image.shape)
    for i in range(np.array(nms_boxes).shape[0]):
        SlidingWindow_images = image[int(nms_boxes[i][1]):int(nms_boxes[i][3]),int(nms_boxes[i][0]):int(nms_boxes[i][2])]
        images = preprocess_fn(SlidingWindow_images,32,32)
        all_images.append(np.array(images))
    all_images = np.array(all_images)
    threadAll = []
    start=0
    newbox = []
    for i in range(threads):
        if (i==threads-1):
            end = len(all_images)
        else:
            end = start+(len(all_images)//threads)
        in_q = all_images[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q ,newbox,nms_boxes,0.995))
        threadAll.append(t1)
        start=end

    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    
    newbox = np.array(newbox)
    new_nms_box_id = py_nms(np.array(newbox)[:,:5],0.1,mode = 'Union')
    #print("new_nms_box_id:",new_nms_box_id)
    new_nms_box = []
    #for i in range(len(new_nms_box_id)):
    for i in range(len(new_nms_box_id)):
        temp = [newbox[new_nms_box_id[i]][0],newbox[new_nms_box_id[i]][1],newbox[new_nms_box_id[i]][2],newbox[new_nms_box_id[i]][3],newbox[new_nms_box_id[i]][4]]
        new_nms_box.append(temp)

    #print("nms_boxes:\n",nms_boxes)
    new_nms_boxes = np.array(new_nms_box)
    print("Proposal bounding box:\n",nms_boxes)
    print("Ground truth bounding box:\n",new_nms_boxes)
    temp_new_nms_box = [new_nms_boxes[0]]
    temp_new_nms_box = np.array(temp_new_nms_box)
    result_image = draw_face(path,new_nms_boxes,None,1)
    file_name = './result/'+'result'+'.jpg'
    cv2.imwrite(file_name,result_image)
    time2 = time.time()
    timetotal = time2 - time1
    fps = float(runTotal / timetotal)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))

    return



# only used if script is run as 'main' from command line
def main():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images. Default is images')
  ap.add_argument('-b', '--box_image_dir', type=str, default='box_image_dir', help='Path to folder of images. Default is images')
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='model_dir/customcnn.xmodel', help='Path of xmodel. Default is model_dir/customcnn.xmodel')

  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --image_dir : ', args.image_dir)
  print (' --box_image_dir : ', args.box_image_dir)
  print (' --threads   : ', args.threads)
  print (' --model     : ', args.model)

  app(args.image_dir,args.box_image_dir,args.threads,args.model)

if __name__ == '__main__':
  main()

