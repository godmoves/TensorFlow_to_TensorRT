# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/bin/env python -tt
r""" TF-TensorRT integration sample script """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import tensorflow.contrib.tensorrt as trt

import numpy as np
import time
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline
import argparse, sys, itertools,datetime
import json
tf.logging.set_verbosity(tf.logging.INFO)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device

def read_tensor_from_image_file(file_name, input_height=224, input_width=224,
                                input_mean=0, input_std=255):
  """ Read a jpg image file and return a tensor """
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='jpg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50)))
  result = sess.run([normalized,tf.transpose(normalized,perm=(0,3,1,2))])
  del sess

  return result

def getSimpleGraphDef():
  """Create a simple graph and return its graph_def"""
  if gfile.Exists("origgraph"):
    gfile.DeleteRecursively("origgraph")
  g = tf.Graph()
  with g.as_default():
    A = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name="input")
    e = tf.constant(
        [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.],[1.,1.,1.,1.,1.,1.]]]],
        name="weights",
        dtype=tf.float32)
    conv = tf.nn.conv2d(
        input=A, filter=e, strides=[1, 1, 1, 1],dilations=[1,1,1,1], padding="SAME", name="conv")
    b = tf.constant([4., 1.5, 2., 3., 5., 7.], name="bias", dtype=tf.float32)
    t = tf.nn.bias_add(conv, b, name="biasAdd")
    relu = tf.nn.relu(t, "relu")
    idty = tf.identity(relu, "ID")
    v = tf.nn.max_pool(
        idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
    out = tf.squeeze(v, name="resnet_v1_50/predictions/Reshape_1")
    writer = tf.summary.FileWriter("origgraph", g)
    writer.close()
    
  return g.as_graph_def()

def updateGraphDef(fileName):
  with gfile.FastGFile(fileName,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  tf.reset_default_graph()
  g=tf.Graph()
  with g.as_default():
    tf.import_graph_def(graph_def,name="")
    with gfile.FastGFile(fileName,'wb') as f:
      f.write(g.as_graph_def().SerializeToString())
  
def getResnet50():
  with gfile.FastGFile("resnetV150_frozen.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def

def printStats(graphName,timings,batch_size):
  if timings is None:
    return
  times=np.array(timings)
  speeds=batch_size / times
  avgTime=np.mean(timings)
  avgSpeed=batch_size/avgTime
  stdTime=np.std(timings)
  stdSpeed=np.std(speeds)
  print("images/s : %.1f +/- %.1f, s/batch: %.5f +/- %.5f"%(avgSpeed,stdSpeed,avgTime,stdTime))
  print("RES, %s, %s, %.2f, %.2f, %.5f, %.5f"%(graphName,batch_size,avgSpeed,stdSpeed,avgTime,stdTime))

def getFP32(batch_size=128,workspace_size=1<<30):
  trt_graph = trt.create_inference_graph(getResnet50(), [ "resnet_v1_50/predictions/Reshape_1"],
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode="FP32")  # Get optimized graph
  with gfile.FastGFile("resnetV150_TRTFP32.pb",'wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

def getFP16(batch_size=128,workspace_size=1<<30):
  trt_graph = trt.create_inference_graph(getResnet50(), [ "resnet_v1_50/predictions/Reshape_1"],
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode="FP16")  # Get optimized graph
  with gfile.FastGFile("resnetV150_TRTFP16.pb",'wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

def getINT8CalibGraph(batch_size=128,workspace_size=1<<30):
  trt_graph = trt.create_inference_graph(getResnet50(), [ "resnet_v1_50/predictions/Reshape_1"],
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode="INT8")  # calibration
  with gfile.FastGFile("resnetV150_TRTINT8Calib.pb",'wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

def getINT8InferenceGraph(calibGraph):
  trt_graph=trt.calib_graph_to_infer_graph(calibGraph)
  with gfile.FastGFile("resnetV150_TRTINT8.pb",'wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph

def timeGraph(gdef,batch_size=128,num_loops=100,dummy_input=None,timelineName=None):
  tf.logging.info("Starting execution")
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
  tf.reset_default_graph()
  g = tf.Graph()
  if dummy_input is None:
    dummy_input = np.random.random_sample((batch_size,224,224,3))
  outlist=[]
  with g.as_default():
    inc=tf.constant(dummy_input, dtype=tf.float32)
    dataset=tf.data.Dataset.from_tensors(inc)
    dataset=dataset.repeat()
    iterator=dataset.make_one_shot_iterator()
    next_element=iterator.get_next()
    out = tf.import_graph_def(
      graph_def=gdef,
      input_map={"input":next_element},
      return_elements=[ "resnet_v1_50/predictions/Reshape_1"]
    )
    out = out[0].outputs[0]
    outlist.append(out)
    
  timings=[]
  
  with tf.Session(graph=g,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    tf.logging.info("Starting Warmup cycle")
    def mergeTraceStr(mdarr):
      tl=timeline.Timeline(mdarr[0][0].step_stats)
      ctf=tl.generate_chrome_trace_format()
      Gtf=json.loads(ctf)
      deltat=mdarr[0][1][1]
      for md in mdarr[1:]:
        tl=timeline.Timeline(md[0].step_stats)
        ctf=tl.generate_chrome_trace_format()
        tmp=json.loads(ctf)
        deltat=0
        Gtf["traceEvents"].extend(tmp["traceEvents"])
        deltat=md[1][1]
        
      return json.dumps(Gtf,indent=2)
    rmArr=[[tf.RunMetadata(),0] for x in range(20)]
    if timelineName:
      if gfile.Exists(timelineName):
        gfile.Remove(timelineName)
      ttot=int(0)
      tend=time.time()
      for i in range(20):
        tstart=time.time()
        valt = sess.run(outlist,options=run_options,run_metadata=rmArr[i][0])
        tend=time.time()
        rmArr[i][1]=(int(tstart*1.e6),int(tend*1.e6))
      with gfile.FastGFile(timelineName,"a") as tlf:
        tlf.write(mergeTraceStr(rmArr))
    else:
      for i in range(20):
        valt = sess.run(outlist)
    tf.logging.info("Warmup done. Starting real timing")
    num_iters=50
    for i in range(num_loops):
      tstart=time.time()
      for k in range(num_iters):
        val = sess.run(outlist)
      timings.append((time.time()-tstart)/float(num_iters))
      print("iter ",i," ",timings[-1])
    comp=sess.run(tf.reduce_all(tf.equal(val[0],valt[0])))
    print("Comparison=",comp)
    sess.close()
    tf.logging.info("Timing loop done!")
    return timings,comp,val[0],None

def score(nat,trt,topN=5):
  ind=np.argsort(nat)[:,-topN:]
  tind=np.argsort(trt)[:,-topN:]
  return np.array_equal(ind,tind),howClose(nat,trt,topN)

def topX(arr,X):
  ind=np.argsort(arr)[:,-X:][:,::-1]
  return arr[np.arange(np.shape(arr)[0])[:,np.newaxis],ind],ind

def howClose(arr1,arr2,X):
  val1,ind1=topX(arr1,X)
  val2,ind2=topX(arr2,X)
  ssum=0.
  for i in range(X):
    in1=ind1[0]
    in2=ind2[0]
    if(in1[i]==in2[i]):
      ssum+=1
    else:
      pos=np.where(in2==in1[i])
      pos=pos[0]
      if pos.shape[0]:
        if np.abs(pos[0]-i)<2:
          ssum+=0.5
  return ssum/X

def getLabels(labels,ids):
  return [labels[str(x+1)] for x in ids]

if "__main__" in __name__:
  P=argparse.ArgumentParser(prog="test")
  P.add_argument('--FP32',action='store_true')
  P.add_argument('--FP16',action='store_true')
  P.add_argument('--INT8',action='store_true')
  P.add_argument('--native',action='store_true')
  P.add_argument('--num_loops',type=int,default=20)
  P.add_argument('--topN',type=int,default=10)
  P.add_argument('--batch_size',type=int,default=128)
  P.add_argument('--dump_diff',action='store_true')
  P.add_argument('--with_timeline',action='store_true')
  P.add_argument('--workspace_size',type=int,default=1<<10,help="workspace size in MB")
  P.add_argument('--update_graphdef',action='store_true')
  
  f,unparsed=P.parse_known_args()
  print(f)
  valnative=None
  valfp32=None
  valfp16=None
  valint8=None
  res=[None,None,None,None]
  print("Starting at",datetime.datetime.now())
  if f.update_graphdef:
    updateGraphDef("resnetV150_frozen.pb")
  dummy_input = np.random.random_sample((f.batch_size,224,224,3))
  with open("labellist.json","r") as lf:
    labels=json.load(lf)
  imageName="grace_hopper.jpg"
  t = read_tensor_from_image_file(imageName,
                                  input_height=224,
                                  input_width=224,
                                  input_mean=0,
                                  input_std=1.0)
  tshape=list(t[0].shape)
  tshape[0]=f.batch_size
  tnhwcbatch=np.tile(t[0],(f.batch_size,1,1,1))
  dummy_input=tnhwcbatch
  wsize=f.workspace_size<<20
  timelineName=None
  if f.native:
    if f.with_timeline: timelineName="NativeTimeline.json"
    timings,comp,valnative,mdstats=timeGraph(getResnet50(),f.batch_size,
                                     f.num_loops,dummy_input,timelineName)
    printStats("Native",timings,f.batch_size)
    printStats("NativeRS",mdstats,f.batch_size)
  if f.FP32:
    if f.with_timeline: timelineName="FP32Timeline.json"
    timings,comp,valfp32,mdstats=timeGraph(getFP32(f.batch_size,wsize),f.batch_size,f.num_loops,
                                   dummy_input,timelineName)
    printStats("TRT-FP32",timings,f.batch_size)
    printStats("TRT-FP32RS",mdstats,f.batch_size)
  if f.FP16:
    k=0
    if f.with_timeline: timelineName="FP16Timeline.json"
    timings,comp,valfp16,mdstats=timeGraph(getFP16(f.batch_size,wsize),f.batch_size,
                                   f.num_loops,dummy_input,timelineName)
    printStats("TRT-FP16",timings,f.batch_size)
    printStats("TRT-FP16RS",mdstats,f.batch_size)
  if f.INT8:
    calibGraph=getINT8CalibGraph(f.batch_size,wsize)
    print("Running Calibration")
    timings,comp,_,mdstats=timeGraph(calibGraph,f.batch_size,1,dummy_input)
    print("Creating inference graph")
    int8Graph=getINT8InferenceGraph(calibGraph)
    del calibGraph
    if f.with_timeline: timelineName="INT8Timeline.json"
    timings,comp,valint8,mdstats=timeGraph(int8Graph,f.batch_size,
                                   f.num_loops,dummy_input,timelineName)
    printStats("TRT-INT8",timings,f.batch_size)
    printStats("TRT-INT8RS",mdstats,f.batch_size)
  vals=[valnative,valfp32,valfp16,valint8]
  enabled=[(f.native,"native",valnative),
           (f.FP32,"FP32",valfp32),
           (f.FP16,"FP16",valfp16),
           (f.INT8,"INT8",valint8)]
  print("Done timing",datetime.datetime.now())
  for i in enabled:
    if i[0]:
      print(i[1],getLabels(labels,topX(i[2],f.topN)[1][0]))
    
  sys.exit(0)
