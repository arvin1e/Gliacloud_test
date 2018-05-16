import configparser
import csv
import cv2
import datetime
import label_map_util
import numpy as np
import os
import pandas as pd
import socket
import sys
import threading
import time
import tensorflow as tf
from struct import unpack,calcsize
from PIL import Image, ImageDraw, ImageFont

lock = threading.RLock()
class PRED(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		csv_lock = False
		global csv_file
		# Path to frozen detection graph.
		self.PATH_TO_PB = os.path.join('./',Model_path+ '/',Model_File_name)
		self.PATH_TO_LABELS = os.path.join('./',Model_path+ '/',Item_File_name)
		try:
			self.detection_graph = self._load_model()
			self.category_index = self._load_label_map()
		except Exception as e:
			print("[ERROR] model or label map error",e)
			sys.exit(1)
		try:
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.sock = sock
		except(socket.error, msg):
			sys.stderr.write("[ERROR] Sock failed :%s\n" % msg[1])
			sys.exit(1)
		try:
			self.sock.bind((host, port))
			self.sock.listen(listens)
		except Exception as e:
			print('[ERROR] Bind failed :%s\n' % e)
			sys.exit(1)
		#define proto graph
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=Memory_fraction)
		with self.detection_graph.as_default():
			with tf.Session(graph=self.detection_graph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
				pass
		self.listen(sess)

	def listen(self,sess):
		while True:
			print("now is listening ..!")
			client, address = self.sock.accept()
			print("----Client Info----")
			print(client, address)
			#client.settimeout(10)
			threading.Thread(target = self.pred, args = (client,address,sess)).start()
		
	def _load_model(self):
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.PATH_TO_PB, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
		return detection_graph

	def _load_label_map(self):
		label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=ClassNum, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)
		return category_index
	
	#connect, recv, pred and send 
	def pred(self,client,address,sess):
		pack_I = calcsize("!I")
		pack_b = calcsize("!b")
		csv_lock = False
		today = datetime.datetime.now().date() #20XX-0X-XX
		
		global old_ini_time, ini_file, img_y, img_x, channel, Storage_csv, Storage_image, \
		Log_path, Log_result_path, threshold_save
		
		while True:
			
			try:  #detect ini file modify time
				filemt= time.localtime(os.stat(ini_file).st_mtime) 
				ini_time = time.strftime("%Y-%m-%d-%H%M%S",filemt)
				if old_ini_time != ini_time:
					print('ini modified !!')
					
					old_ini_time = ini_time
					#load ini
					try:
						#envir info
						host = config.get('HostIP', 'IP')
						port = int(config.get('HostIP', 'PortNo'))
						dim = 1
						
						#img size
						img_x = int(config.get('Resize_Img','img_x'))
						img_y = int(config.get('Resize_Img','img_y'))
						channel = int(config.get('Resize_Img','channel'))
						
						Rimg = np.array(config.items('Resize_Img'))
						for i in Rimg[1:,1]:
							dim *= int(i)
						
						DEVICES = config.get('HostIP', 'CUDA_VISIBLE_DEVICES')
						os.environ['CUDA_VISIBLE_DEVICES'] = DEVICES
						
						print("HostIP =", host,"| Port =", port, "| Dim =",dim, '| GPU usage NO =',str(DEVICES)) 
						
						#Other 
						Memory_fraction = float(config.get('Other','Memory_fraction'))
						Show_message = config.getboolean('Other','Message')
						Storage_image = config.getboolean('Other','Storage_image')
						Storage_csv = config.getboolean('Other','Storage_csv')
						Cut_Level = float(config.get('Other','Cut_Level'))
						threshold_save = float(config.get('Other','threshold_save'))
						CSV_Line = int(config.get('Other','CSV_Line'))

						print("Memory_fraction =", Memory_fraction,"| Show_message =", Show_message, "| Storage_image =",Storage_image,
								"| Storage_csv =",Storage_csv, '| Cut_Level =',Cut_Level ,'| CSV_Line =',CSV_Line) 
						
						#Socket
						Model_path = config.get('Socket','Model_path')
						Model_File_name = config.get('Socket','Model_File_name')
						Item_File_name = config.get('Socket','Item_File_name')
						ClassNum = int(config.get('Socket','ClassNum'))
						listens = int(config.get('Socket','Listen'))
						Log_path = config.get('Socket','Log_path')
						WaitTimeOut = config.get('Socket','WaitTimeOut')
						Recv_size = config.get('Socket','Recv_Size')
						
						print("Model_path =", Model_path,"| Model_File_name =", Model_File_name, "| Item_File_name =",Item_File_name
								, '| ClassNum =',ClassNum, '| Log_path =',Log_path) 
						#for test input path
						Input_path = config.get('InputData','Path')
						
						#create date folder path
						date=datetime.datetime.now().date()
						date = str(date)	
						date = date.replace('-','')
						
						# create csv log and Pic output path by date
						if not os.path.exists(Log_path+'/'+date+'/Log'):
							os.makedirs(Log_path+'/'+date+'/Log')
							os.makedirs(Log_path+'/'+date+'/Pic')
						# show output CSV result path
						Log_result_path = Log_path+'/'+date+'/Log/'
						print('Log_path =',Log_path,'Log_result_path =',Log_result_path)
					except Exception as e:
						print("[ERROR]",e)
						sys.exit(1)
			except Exception as e:
				print("[ERROR] ini reload error :",e)
			
			
			#start receive data--------
			with lock:
				Package_time = time.time()
				try:
					print("----recv data----")
					msg1 = client.recv(2**15)
					total_len = int.from_bytes(msg1[0:4],byteorder='big')
					while len(msg1)<total_len:
						msg2 = client.recv(2**15)
						msg1+=msg2
					#cope with package--------
					Predict_time = time.time()
					
					pkg = msg1
					print("msg1",len(msg1))
					print("----after packing----")
					#print("pkg len:",len(pkg))
					#file_len = int.from_bytes(pkg[0:pack_I],byteorder='big')
					file_name_len = int.from_bytes(pkg[pack_I:pack_I+pack_b],byteorder='big')
					file = int.from_bytes(pkg[pack_I+pack_b:],byteorder='big')
					
					_,_,file_name,msg1 = unpack("!Ib%ds%ds" % (file_name_len,total_len-file_name_len-pack_I-pack_b,),pkg)
					print("file_name:",file_name.decode("utf8"))
					#print("msg1:",np.array(bytearray(msg1)))
					idx = file_name.decode("utf8").index(".")
					File = file_name.decode("utf8")[:idx]
					
					#msg = np.asarray(bytearray(msg1))
					msg = np.fromstring(msg1, np.uint8)
					msg = msg.reshape(img_y,img_x,channel)
					
				except Exception as e:
					print("[ERROR] now is disconnected :",e)
					break
				try:
					# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
					img = np.expand_dims(msg, axis=0)
					
					image_tensor = self.detection_graph.get_tensor_by_name('in/X:0')
					soft = self.detection_graph.get_tensor_by_name('out/Softmax:0')
					'''
					image_tensor = self.detection_graph.get_tensor_by_name('in/X:0')
					soft = self.detection_graph.get_tensor_by_name('FullyConnected_2/Softmax:0')
					'''
					print("----predict----")
					soft = sess.run(soft,feed_dict={image_tensor: img})
					Predict_time = round(time.time()-Predict_time,6)
					print("prob:",soft[0,0])
					if soft[0,0]>=Cut_Level:
						ansr = 'NG'
					else:
						ansr = 'OK'
					print("predict class:",ansr)
					client.send(bytes(ansr.encode('utf-8')))
					Package_time = round(time.time()-Package_time,6)
					
					date=datetime.datetime.now().date()
					timestamp = time.strftime("_%H%M%S",time.localtime())
					
					
					if Storage_csv:
						row = np.array([File+timestamp, str(address), ansr, \
						soft[0][np.argmax(soft)],str(Predict_time),str(Package_time)]).reshape(1,6)
						
						# create csv log output path by date
						date2 = str(date)	
						date2 = date2.replace('-','')
						if not os.path.exists(Log_path+'/'+date2+'/Log'):
							os.makedirs(Log_path+'/'+date2+'/Log')
						Log_result_path = Log_path+'/'+date2+'/Log/'
						
						#whether output header or not
						if not os.path.isfile(Log_result_path+date2+"_result.csv"):
							cols = ['FileName', 'Address', 'Class', 'Score', 'PredictTime','PackageTime']
							df = pd.DataFrame([]*6,columns = cols)
							
							df.to_csv(Log_result_path+date2+"_result.csv",header = True,index=False)
						else:
							df = pd.DataFrame(row)
							df.to_csv(Log_result_path+date2+"_result.csv",mode='a',header = False,index=False)			
				except Exception as e:
					print("[ERROR] CSV log error :",e)
				
				#store picture exceeding threshold_save
				try:
					if Storage_image :
						date=datetime.datetime.now().date()
						date = str(date)	
						date = date.replace('-','')
						# create csv log output path by date
						if not os.path.exists(Log_path+'/'+date+'/Pic'):
							os.makedirs(Log_path+'/'+date+'/Pic')
						Pic_result_path = Log_path+'/'+date+'/Pic/'
						thres_valid = soft[0]>threshold_save
						#print(np.any(thres_valid))
						if np.any(thres_valid):
							#msg_soft(larger than threshold_save)
							print("----saving image----")
							pic = Image.fromarray(msg)
							pic = pic.convert('RGB')
							pic.save(Pic_result_path+file_name.decode("utf8"))
							print("----image saved----")
				except Exception as e:
					print("[ERROR] Pic log error:",e)

if __name__ == '__main__':	
	global old_ini_time,ini_file
	try:
		if len(sys.argv) < 2 or len(sys.argv) >2:
			print("type in: python XX.py YY.ini!")
			sys.exit(1)
		print("input ini file:",sys.argv[1])
		#ini_file = 'AI_Config.ini'
		ini_file = sys.argv[1]
		if ini_file[-4:]==".ini" and os.path.isfile(ini_file):
			config = configparser.ConfigParser()
			config.read(ini_file)
		else:
			print("please enclosing one accurate ini file!")
			sys.exit(1)
	except:
		sys.exit(1)
	
	#envir info
	host = config.get('HostIP', 'IP')
	port = int(config.get('HostIP', 'PortNo'))
	#load ini Other 
	Memory_fraction = float(config.get('Other','Memory_fraction'))
	#Socket
	Model_path = config.get('Socket','Model_path')
	Model_File_name = config.get('Socket','Model_File_name')
	Item_File_name = config.get('Socket','Item_File_name')
	ClassNum = int(config.get('Socket','ClassNum'))
	listens = int(config.get('Socket','Listen'))
	
	filemt = time.localtime()
	old_ini_time = time.strftime("%Y-%m-%d-%H%M%S",filemt)

	# start Socket and Predict
	detect = PRED()


