#from google.colab import drive  # google.colab.drive를 drive.으로 사
#drive.mount('/content/drive') # 경로고정/ 이후에는 /content/drive/MyDrive/ drive 내에 경로지정

import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader,random_split # torch.utils.data =데이터학습용 라이브러리

#import os
#import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# python -> matlab 데이터보내기
import socket # 통신 library
import time # 아마 time.sleep 함수 쓰려고


device = "cuda" if torch.cuda.is_available() else "cpu"

#모델복붙
class SignalCNN(nn.Module): # 예시로 nn.Module,parameters() 가있는데 이걸 SignalCNN.parameters로 사용해서 내가만든 모델의 가중치에 접근&관리 하기 쉽게해줌
  def __init__(self,input_channels,output_dim): # self:자기자신을 가르키는 파이썬기능   model. 이렇게 사용하려면 def하고 첫번째인자로 self를 받아야함
    super().__init__() # 부모클래스의 초기설정을 실행하는건데 pytorch 머신러닝 국룰이래
    # SignalCNN을 선언하면 다음의 함수를 만든다 .conv1 + .BatchNorm1d + .ReLU +  .MaxPool1d + .AdaptiveAvgPool1d + .Flatten + .Linear
    self.conv1=nn.Conv1d(input_channels,16,kernel_size=11,padding=5) # 입력신호의채널
                                                                    # ,필터갯수=추출할특징의갯수(필터갯수에 맞춰 필터를 랜덤으로일단만들고 입력전체채널에 콘볼루션연산수행)
                                                                    # kernel_size: 필터샘플길이
                                                                    # padding : 머신러닝 conv연산은 신호바깥을 0으로 취급하지않아서 앞뒤 000 padding해줘서 출력신호길이 맞추고싶을때
                                                                    # (출력길이: 입력-커널+1)
    #BatchNorm1d: 출력채널의 숫자들을 평균과분산 맞춰준다(z-score정규화랑비슷)
    self.bn1=nn.BatchNorm1d(16)
    # ReLu : conv연산한 값들중에 음수값들전부 0으러 바꾼다
    # 왜? 필터란 특정패턴을 찾기위한 값배열들 그래서 특정패턴을찾는 필터 =양수가 크게나와야 해서 음수는 0으로처리한데
    self.relu1=nn.ReLU()
    # Conv 값들을 kernel_size로 묶어서 큰값만추출= 샘플길이줄임 stride=kernel_size로 맞춰줘서 겹치지않게
    self.pool1=nn.MaxPool1d(kernel_size=2,stride=2) # /7

    # 2layer conv1이 짧은 샘플구간의 특징을 추출하면 2번째레이어가 그특징의 그특징을 추출한데: 반복되는가? 구조?
    # 3layer는 그특징의 특징의 특징 :전체적인신호특징
    # 3layer가 무난해?
    self.conv2=nn.Conv1d(16,32,kernel_size=9,padding=4)
    self.bn2=nn.BatchNorm1d(32)
    self.relu2=nn.ReLU()
    self.pool2=nn.MaxPool1d(kernel_size=2,stride=2) # /14



    self.conv3=nn.Conv1d(32,64,kernel_size=7,padding=3)
    self.bn3=nn.BatchNorm1d(64)
    self.relu3=nn.ReLU()
    self.pool3=nn.MaxPool1d(kernel_size=2,stride=2) # /7/14/
    self.drop3=  nn.Dropout1d(0.1)

    self.conv4=nn.Conv1d(64,128,kernel_size=5,padding=2)
    self.bn4=nn.BatchNorm1d(128)
    self.relu4=nn.ReLU()
    self.pool4=nn.MaxPool1d(kernel_size=2,stride=2) #
    self.drop4=  nn.Dropout1d(0.15)


    #self.conv5=nn.Conv1d(128,256,kernel_size=3,padding=1)
    #self.bn5=nn.BatchNorm1d(256)
    #self.relu5=nn.ReLU()
    #self.drop5= nn.Dropout1d(0.2)
    #conv연산으로 특징 추출한 뒤에는  각채널당 값을 1개로요약
    # 그래서 필터로 추출할 특징이 얼마나 강하게 나타났는가? 만 요약  시간축값은 사라짐(10으로늘리거나 추가로다른모델 사용가능하다..)
    # mean써도 되긴한다는데 .. 음.. (1차원으로바로)
    avg=32
    self.global_pool = nn.AdaptiveAvgPool1d(avg) #아직3차원이래 만약8이면채널마다 최종8개 값
    # 마지막차원을 없앤다.. 왜?Linear는 2차원입력을기대한다고함
    self.flatten=nn.Flatten() # channel*최종값 = 최종값들을 1차원으로펼침
    self.dropout=nn.Dropout(0.3) # 과적합 방지
    # 이제뽑아낸 최종필터갯수(특징) 값들 보고 최총 예측값(dim=?)으로 바꾸는 함수.Linear
    # y=w*x + bias
    # 16*1= w * (64*1) +bias
    # w=(16*64) 가되어야하네?
    # 처음w,b는  랜덤초기화
    self.fc=nn.Linear(128*avg,output_dim)
  def forward(self,x): # 이거이름바꾸면안된대 pytorch 에서는 nn.Module을 상속받으면 자동으로 forward가 실행된다..
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.pool1(x)


    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.pool2(x)


    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu3(x)
    x = self.pool3(x)
    x = self.drop3(x)

    x = self.conv4(x)
    x = self.bn4(x)
    x = self.relu4(x)
    x = self.pool4(x)
    x = self.drop4(x)

    #x = self.conv5(x)
    #x = self.bn5(x)
    #x = self.relu5(x)
    #x = self.drop5(x)

    x = self.global_pool(x)
    x = self.flatten(x)
    x = self.dropout(x)
    x = self.fc(x)

    return x

angle_table=np.array([[90,-20+40,-90+30,-90+60,-70+60,-70+20,-70+20,-70+70,-70+20,-70+20,-60+70,-60+20,-80+20,-70+70,-70+20,-70+20],
                     [0,-20+70,-90+30,-90+30,-70+70,-70+20,-70,-70+40,-70+40,-70+10,-60+60,-60+20,-80,-70+80,-70+5,-70+10],
                     [90,-20+90,-90+10,-90+10,-70+90,-70+80,-70+30,-70+90,-70+90,-70+20,-60+90,-60+70,-80+30,-70+90,-70+70,-70+30]],dtype=np.float32) # 뒤에 dtype?

# test 파일저장load
# def load_mat_dataset(folder_path,idx):
#   X_list=[] # 누적하려면 list가좋다
#   Y_list=[]
#   y=[0,0,2,2,2,0,2,2,0,2,1,1,0,0,2,2,2,1,1,2,0,0,2,2,1,1,2,2,2,1,0,0,1,0,1,1,0,1,0,2,2,2,1,1,1,2,1,0,1,2,1,0,2,2,0,2,1,2,0,0,0,1,1,1,0,0,0,0,0,1,2,2,0,0,1,2,1,2,0,1,0,1,1,0,0,2,2,2,2,0,0,2,2,0,1,0,0,0,0,2,1,2,1,1,1,2,1,0,1,2,0,0,1,0,1,1,1,2,0,1,2,1,0,1,1,2,2,0,1,2,1,0,0,0,2,2,1,1,1,2,2,0,2,0,1,1,1,0,2,2,2,0,2,0,0,0,1,1,2,2,0,2,1,2,0,0,2,0,2,0,0,1,0,2,0,1,2,0,2,2,1,1,0,2,2,0,2,2,0,1,1,0,2,1,2,2,0,2,1,1,2,0,1,0,2,0,1,1,0,1,2,2,1,1,0,1,2,0,1,0,1,0,0,1,2,1,2,2,1,1,2,1,2,2,0,2,1,1,1,2,0,0,0,1,2,1,2,0,1,0,1,2,0,2,2,2,0,1,0,0,2,0,0,1,1,0,1,1,0,0,2,1,1,0,1,1,2,0,2,2,0,1,2,2,2,1,0,1,0,1,1,0,1,2,0,1,2,2,0,1,2,1,1,2,1,1,0,0,1,1,0,1,1,0,1,0,0,2,1,2,1,2,1,0,2,2,0,1,0,1,0,2,2,2,0,0,2,0,2,1,2,2,1,1,2,2,2,2,0,0,1,2,0,2,1,0,1,2,0,1,0,2,0,2,1,0,2,2,1,1,2,2,1,0,1,1,2,2,1,1,0,2,0,0,1,0,1,1,0,2,1,1,2,1,0,0,1,0,1,2,2,2,1,0,0,1,0,0,2,0,1,0,2,1,1,2,0,0,2,2,2,2,0,0,0,0,0,2,0,2,2,1,1,0,2,1,1,0,2,2,1,0,1,0,1,1,2,0,2,0,0,0,0,0,2,0,2,2,1,2,2,1,0,1,1,0,0,0,0,2,0,0,1,1,0,2,2,1,0,1,1,0,1,1,1,2,0,0,0,2,2,2,1,1,2,1,1,0,0,2,0,2,1,2,1,1,1,2,0,1,2,2,1,0,1,2,2,1,1,1,2,2,1,0,0,0,0,2,2,0,0,1,0,1,2,0,0,0,2,1,0,2,1,0,0,1,1,2,2,0,0,2,2,1,2,0,1,2,0,0,1,2,2,2,2,1,1,2,0,1,0,2,2,2,1,1,0,1,0,1,2,2,0,1,1,1,0,2,2,2,1,2,0,0,1,1,2,0,1,2,1,1,2,0,2,1,1,2,1,0,1,0,0,0,2,0,1,0,1,1,2,2,2,1,0,0,1,1,0,0,1,2,0,2,0,1,2,2,0,2,1,0,2,0,1,1,0,2,1,0,1,1,1,1,0,0,1,2,2,2,2,0,1,0,0,1,1,2,1,1,0,0,0,0,1,2,0,0,2,0,2,1,1,2,0,2,0,0,2,2,1,2,1,2,0,1,0,2,1,2,0,2,1,0,0,1,2,0,1,2,2,2,0,2,0,2,0,2,1,1,2,2,1,1,2,0,0,1,1,1,0,0,1,2,0,0,2,2,1,1,2,2,2,1,0,1,2,0,2,2,2,0,2,0,0,0,1,1,2,2,0,2,1,2,0,0,2,0,2,0,0,1,0,2,0,1,2,0,2,2,1,1,0,2,2,0,2,2,0,1,1,0,2,1,2,2,0,2,1,1,2,0,1,0,2,0,1,1,0,1,2,2,1,1,0,1,2,0,1,0,1,0,0,1,2,1,2,2,1,1,2,1,2,2,0,2,1,1,1,2,0,0,0,1,2,1,2,0,1,0,1,2,0,2,2,2,0,1,0,0,2,0,0,1,1,0,1,1,0,0,2,1,1,0,1,1,2,0,2,2,0,1,2,2,2,1,0,1,0,1,1,0,1,2,0,1,2,2,0,1,0,0,0,0,2,0,2,2,1,2,2,1,0,1,1,0,0,0,0,2,0,0,1,1,0,2,2,1,0,1,1,0,1,1,1,2,0,0,0,2,2,2,1,1,2,1,1,0,0,2,0,2,1,2,1,1,1,2,0,1,2,2,1,0,1,2,2,1,1,1,2,2,1,0,0,0,0,2,2,0,0,1,0,1,2,0,0,0,2,1,0,2,1,0,0,1,1,2,2,0,0,2,2,1,2,0,1,2,0,0,1,2,2,2,2,1,1,2,0,1,0,2,2,2,1,1,0,1,0,1,2,2,0,1,1,1,0,2,2,2,1,2,0,0,1,1,2,0,1,2,2,0,2,0,0,0,1,1,2,2,0,2,1,2,0,0,2,0,2,0,0,1,0,2,0,1,2,0,2,2,1,1,0,2,2,0,2,2,0,1,1,0,2,1,2,2,0,2,1,1,2,0,1,0,2,0,1,1,0,1,2,2,1,1,0,1,2,0,1,0,1,0,0,1,2,1,2,2,1,1,2,1,2,2,0,2,1,1,1,2,0,0,0,1,2,1,2,0,1,0,1,2,0,2,2,2,0,1,0,0,2,0,0,1,1,0,1,1,0,0,2,1,1,0,1,1,2,0,2,2,0,1,2,2,2,1,0,1,0,1,1,0,1,2,0,1,2,2,0,1,0,0,0,0,2,0,2,2,1,2,2,1,0,1,1,0,0,0,0,2,0,0,1,1,0,2,2,1,0,1,1,0,1,1,1,2,0,0,0,2,2,2,1,1,2,1,1,0,0,2,0,2,1,2,1,1,1,2,0,1,2,2,1,0,1,2,2,1,1,1,2,2,1,0,0,0,0,2,2,0,0,1,0,1,2,0,0,0,2,1,0,2,1,0,0,1,1,2,2,0,0,2,2,1,2,0,1,2,0,0,1,2,2,2,2,1,1,2,0,1,0,2,2,2,1,1,0,1,0,1,2,2,0,1,1,1,0,2,2,2,1,2,0,0,1,1,2,0,1,2,2,0,1,2,1,1,1,0,0,0,1,2,0,0,2,2,2,1,2,0,1,0,2,0,1,1,0,2,0,1,1,1,2,0,0,2,2,0,0,0,0,0,1,1,2,2,0,1,1,2,0,0,0,0,2,2,2,1,1,0,1,2,0,2,2,2,1,1,1,0,1,1,2,2,0,0,2,1,2,2,1,2,0,0,0,2,1,2,0,1,0,1,2,2,2,1,0,1,1,0,2,1,0,2,0,2,1,1,2,1,1,1,1,1,0,1,1,2,2,1,1,0,0,1,0,0,2,2,1,1,2,2,0,1,2,2,0,0,0,2,2,0,2,2,1,0,1,0,2,0,0,0,1,1,1,0,2,2,1,2,1,2,2,0,0,1,0,0,2,1,1,2,2,0,1,1,1,2,2,2,0,1,1,1,1,1,2,0,2,2,0,2,2,0,0,2,2,0,2,0,0,2,2,1,0,0,1,2,2,0,1,1,1,0,2,0,0,0,0,0,0,1,0,2,2,0,0,0,2,1,0,2,0,2,2,1,0,1,1,2,1,1,1,0,2,1,1,1,2,1,0,0,0,2,1,2,1,0,2,1,2,2,2,2,2,1,0,2,2,1,1,0,0,0,1,0,2,0,1,1,0,1,1,2,0,1,1,0,2,2,0,1,0,2,0,2,1,1,2,1,1,0,0,2,0,1,1,1,0,0,1,0,0,1,1,2,1,1,1,2,1,2,2,1,1,1,0,1,0,0,2,0,1,0,2,0,2,0,2,1,0,1,2,2,0,0,1,1,1,0,1,2,2,2,1,1,1,2,2,1,2,0,2,1,1,0,2,0,0,2,2,1,1,0,2,2,2,1,2,0,0,2,1,2,0,1,1,0,2,0,2,1,0,1,2,1,1,2,0,1,1,2,2,0,0,1,1,0,2,2,0,0,1,2,0,2,0,2,1,2,0,0,0,0,2,1,1,2,0,1,2,2,2,0,2,0,0,2,0,0,2,0,1,0,1,2,2,0,1,2,0,2,1,2,2,0,0,0,0,1,0,2,2,1,0,1,0,2,2,0,1,2,0,1,0,1,2,1,1,1,1,1,2,0,1,2,0,1,2,1,1,0,1,1,0,1,1,2,2,0,1,0,2,0,0,1,0,2,0,1,0,1,1,1,0,1,0,0,1,1,2,2,0,1,2,0,2,0,2,2,2,1,0,0,2,1,2,0,1,0,2,1,1,1,1,2,2,1,2,0,1,0,0,2,2,2,0,2,1,2,2,2,1,0,2,0,0,2,0,0,0,2,1,2,1,0,0,1,2,1,2,2,2,0,0,2,2,1,2,1,1,0,1,0,0,0,2,2,1,2,0,1,0,1,2,1,0,2,1,0,2,0,0,0,0,2,2,1,0,2,2,2,0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,1,2,1,0,2,1,0,0,2,1,1,0,1,1,2,0,1,0,2,1,1,1,2,2,1,0,1,0,2,2,2,0,0,1,0,1,2,0,2,2,0,0,0,1,1,0,0,2,0,1,2,0,0,0,2,1,0,2,1,2,2,1,2,2,0,2,1,1,2,2,1,0,2,0,2,2,1,2,1,1,1,2,1,1,2,1,2,1,2,1,0,1,0,1,2,2,2,0,2,1,2,1,2,1,0,1,2,2,1,2,0,2,1,2,2,1,2,0,1,2,0,0,0,2,2,0,1,2,1,1,1,0,1,2,0,2,2,2,1,0,0,1,2,2,2,0,0,2,2,2,2,0,1,2,2,2,2,2,1,2,1,0,0,1,1,2,2,0,0,1,2,1,2,1,1,2,0,0,2,0,2,1,2,1,1,1,0,0,0,2,2,0,0,1,1,1,0,1,1,2,0,1,2,1,0,1,0,0,1,1,1,0,2,0,0,1,0,2,1,1,0,0,2,1,0,0,0,0,2,2,0,0,0,2,1,1,2,0,0,2,0,1,1,0,0,1,1,2,0,1,1,2,2,1,1,1,2,1,0,2,2,0,1]
#   for i in idx:
#      path = os.path.join(folder_path, f"sample{i:02d}.mat") # sample{i} 대산 sample{i:02d}로해서 01 02 03 이렇게표현
#      mat=sio.loadmat(path,struct_as_record=False, squeeze_me=True)
#      epo=mat["epo"] # epo는 내 matlab에서 이름이 epo임 ..
#      x=epo.x
#      x=np.transpose(x,(2,1,0))
#      x=x.astype(np.float32)
#      #z-score 정규화 axis 2가 시간축
#      mean=np.mean(x,axis=2,keepdims=True)
#      std=np.std(x,axis=2,keepdims=True) + 1e-6
#      x=(x-mean)/std
#      # 라벨정리
#      #class_label=y
#      #angle_label=angle_table[class_label]
#      X_list.append(x) # x값을 X_list에 계속누적 누적 sample1: 1~150 trial먼저누적  다음 151~300 누적
#      #Y_list.append(angle_label) # Y_list : 총 trial(2250)  * 16 (angle각)
#      print(f"{path}")
#      print("x:",x.shape)
#   # concatenate: list 안에있는 numpy array 변수를 하나로합칠떄..
#   X=np.concatenate(X_list,axis=0) # list 를 np용 array로만들면서 한번에 합침
#                                   # np.cocatenate a로 바로하면 [X x] 이런식으로 계속 누적해줘야해서 비효율적이래  마지막
#   Y=angle_table[y]
#   return X,Y

# ## --- 여기는 한번저장햇기때문에 스킵
# test_X,test_Y=load_mat_dataset("/content/drive/MyDrive/Test set",[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
# # .mat파일 .npz로 저장
# np.savez("/content/drive/MyDrive/test_signal_data.npz",test_X=test_X,test_Y=test_Y)
data = np.load("C:/Users/juwon/Desktop/MatlabSimpscape/test_signal_data.npz")

#뽑을 data size
batch=1

test_X = data["test_X"]
test_Y = data["test_Y"]
print("test_x ",test_X.shape)
print("test_y ",test_Y.shape)
X_tensor=torch.from_numpy(test_X).float() #.double or float or half or int or long or byte...
Y_tensor=torch.from_numpy(test_Y).float()
test_loader=DataLoader(TensorDataset(X_tensor,Y_tensor),batch_size=batch,shuffle=True)
#모델선언
model=SignalCNN(60,16).to(device)
checkpoint=torch.load("C:/Users/juwon/Desktop/MatlabSimpscape/model.pth",map_location=device)

model.load_state_dict(checkpoint) #  모델 가중값들 불러온다.

# test모드
model.eval()

# test하기
HOST="127.0.0.1" #=자기자신
PORT=5000  # 아무값

#통신용 객체만들기 .socket(주소방식v4 or v6 ,통신방식 tcp or udp)
server=socket.socket(socket.AF_INET,socket.SOCK_STREAM) # v4 & tcp 선택
server.bind((HOST,PORT)) # 호스트 컴퓨터의 주소+포트 를 사용해서 서버로 만듬 
server.listen(1) # 접속요청을 들을 준비가 되었다
print("MATLAB 접속 대기중...")
conn,addr=server.accept() #.accept:접속대기 conn:통신용객체 ,addr:수신받은 주소?정
print("MATLAB 접속 완료 : ",addr)





test_loss=0.0
criterion=nn.MSELoss()  #객체만들기
test_number=0
tot_number=0

with torch.no_grad():
  for batch_X,batch_Y in test_loader:
    tot_number+=1
    batch_X=batch_X.to(device)
    batch_Y=batch_Y.to(device)
    pred=model(batch_X)
    #loss=criterion(pred,batch_Y)

    # loss계산 스킵 test_loss+=loss.item()*batch_X.size(0) # batch만큼 곱해
    
  

    #내가 답아니깐 70이랑 40초과하면넘기고 아니면스킵하고
    # test_numer+1
    if (pred[0][0]<40 and batch_Y[0][0]==0) or (pred[0][0]>60 and batch_Y[0][0]==90):
      test_number+=1
      print("관절예측",pred)
      print("정답 자세 :",batch_Y)
      values=pred[0].detach().cpu().numpy().astype('float32')
      conn.sendall(values.tobytes())
      time.sleep(0.1)
    else:
      continue
    

   


    # >0 =1 회  >4 =5회 반복
    if test_number>4:
      print("종료: ",test_number)
      print("총반복: ",tot_number)
      break



test_loss= test_loss/(test_number*batch)

print("test_loss: ",test_loss) # 평균20도차이나네
# 이제 pred 값을 matlab으로 넘기는거 해보자
# pred 5번해보고 ㅇㅇ
