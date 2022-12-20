from dataclasses import dataclass
from matplotlib import animation
from scipy.interpolate import interp1d
import imufusion
import matplotlib.pyplot as pyplot
import numpy
import modi
import time
import pandas as pd
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def HPF2(series, cutoff,fs, order=1):
    '''
    series : 데이터
    high : 최고 구간(0< low < 1)
    order : 필터 계수, 높을수록 민감
    '''
    nyq = 0.5*fs
    cutoff = cutoff/nyq
    
    b, a = butter(
                  N = order,
                  Wn = cutoff,
                  btype = 'high',
                  )
    hpf_series =  signal.lfilter(b, a, series)
    
    return hpf_series


def LPF(series, cutoff,fs, order=3):
    '''
    series : 데이터
    low : 최저 구간(0< low < 1)
    order : 필터 계수, 높을수록 민감
    '''
    nyq = 0.5*fs
    cutoff = cutoff/nyq
    
    b, a = butter(
                  N = order,
                  Wn = cutoff,
                  btype = 'low',
                  )
    lpf_series = signal.filtfilt(b, a, series)
    
    return lpf_series

def HPF(series, cutoff,fs, order=3):
    '''
    series : 데이터
    high : 최고 구간(0< low < 1)
    order : 필터 계수, 높을수록 민감
    '''
    nyq = 0.5*fs
    cutoff = cutoff/nyq
    
    b, a = butter(
                  N = order,
                  Wn = cutoff,
                  btype = 'high',
                  )
    hpf_series =  signal.filtfilt(b, a, series)
    
    return hpf_series

def calib_gyro(gyro):
    acc_x=[0]
    acc_y=[0]
    acc_z=[0]
    i=0
    df = pd.DataFrame([[datetime.datetime.now(),0,0,0,0,0,0]], columns = ['timestamp','acc_x','acc_y','acc_z', 'ang_x','ang_y', 'ang_z'])
    print("gyro calibration...")
    while True:
        acc_x = (gyro.acceleration_x)/50
        acc_y = (gyro.acceleration_y)/50
        acc_z = (gyro.acceleration_z)/50
        ang_x = gyro.angular_vel_x*3.32
        ang_y = gyro.angular_vel_y*3.32
        ang_z = gyro.angular_vel_z *3.32
        df.loc[len(df)] = [datetime.datetime.now(),acc_x, acc_y,acc_z,ang_x,ang_y,ang_z]
        i +=1
        if i>2000:
            print("calibration done")
            break

    calib_acc_x = df['acc_x'].mean()
    calib_acc_y = df['acc_y'].mean()
    calib_acc_z = df['acc_z'].mean()
    calib_ang_x = df['ang_x'].mean()
    calib_ang_y = df['ang_y'].mean()
    calib_ang_z = df['ang_z'].mean()
    result = [calib_acc_x,calib_acc_y,calib_acc_z, calib_ang_x,calib_ang_y,calib_ang_z]
    return result

def preprocess(feature_df):
    df = feature_df.drop(['timestamp_mod'], axis = 1)
    #for i in df.columns[:12]:
    #    df[i] = (df[i].diff().fillna(0))
        
    return df

def pred(df, model):
    pred_r = model.predict(df)
    
    pred_df = pd.DataFrame(df, columns = df.columns)
    pred_df.reset_index(drop=True, inplace= True)
    pred_df['is_moving'] = pred_r
    
    return pred_df

def train_preprocess(train_df):
    train_df.drop(['timestamp_mod'], axis = 1 , inplace = True)
    y = train_df.label
    train_df.drop(['label'], axis = 1, inplace = True)
    train_df.to_csv('testdata.csv')
    '''
    # 시간간격 0.1초
    #train_df.drop(['Unnamed: 0'], axis = 1 , inplace = True)
    train_df['timestamp_f'] = train_df['timestamp_mod'].apply(lambda x : np.round(x, 1))
    train_df_sec = train_df.drop_duplicates(['timestamp_f'], keep='first')
    train_df_sec.drop(['timestamp_mod'], axis = 1, inplace = True)
    
    # 변화량
    y = train_df_sec.label
    train_df_sec.drop(['label', 'timestamp_f'], axis = 1, inplace = True)
    train_df_diff = pd.DataFrame(columns = train_df_sec.columns)
    for i in train_df_sec.columns[:12]:
        train_df_diff[i] = (train_df_sec[i].diff().fillna(0))
    train_df_diff.reset_index(drop = True, inplace=True)
    train_df_diff.to_csv('testdata.csv')
    
    return train_df_diff, y
    '''
    return train_df, y

def train(feature_df, label):
    X_train, X_test, y_train, y_test = train_test_split(feature_df, label, test_size = 0.2, random_state = 5, stratify = label) # stratify 수정
    feature_df.to_csv('traindata.csv')
    # 모델 불러오기
    model = RandomForestClassifier()

    # 학습
    model.fit(X_train, y_train)
    
    return model



class Light():
    def __init__(self, interval):
        self.interval = interval
        self.light = modi.MODI(
        #conn_type='ble',
        #network_uuid="0x7fbcd062a760"
            )
        self.network = self.light.networks[0]
        self.gyro=self.light.gyros[0]
        self.network = self.light.networks[0]
        self.env = self.light.envs[0]
        self.speaker= self.light.speakers[0]
        self.button = self.light.buttons[0]
        self.led = self.light.leds[0]

        self.gyrodf = pd.DataFrame([[datetime.datetime.now(),0,0,0,0,0,0]], columns = ['timestamp','ang_x','ang_y','ang_z', 'acc_x','acc_y', 'acc_z'])
        self.envdf = pd.DataFrame([[datetime.datetime.now(),0,0,0]], columns = ['timestamp' ,'brightness','humidity','temperature'])
        self.calib_data = calib_gyro(self.gyro)

    def get_train_model(self):  # 워킹 데이터 학습
        self.df = pd.DataFrame([[0,0,0,0,0,0,0,0]], columns = ['timestamp','ang_x', 'ang_y','ang_z','acc_x','acc_y','acc_z','label'])
        self.start = datetime.datetime.now()
        self.end = self.start+datetime.timedelta(seconds=10)
        self.now = datetime.datetime.now()
        print('데이터 수집,,, 걸을때 버튼을 누르세요')
        print("데이터 수집 종료 : 더블클릭")
        while True:
    
            self.acc_x = ((self.gyro.acceleration_x)/48.5)-self.calib_data[0]
            self.acc_y = ((self.gyro.acceleration_y)/48.5)-self.calib_data[1]
            self.acc_z = ((self.gyro.acceleration_z)/48.5)-self.calib_data[2]

            self.ang_x = self.gyro.angular_vel_x*3.32-self.calib_data[3]
            self.ang_y = self.gyro.angular_vel_y*3.32-self.calib_data[4]
            self.ang_z = self.gyro.angular_vel_z *3.32-self.calib_data[5]
            if self.button.pressed:
                self.df.loc[len(self.df)] = [(str(datetime.datetime.now()-self.start)),self.ang_x, self.ang_y, self.ang_z, self.acc_x, self.acc_y,self.acc_z,1]
                print("press")
            else :
                self.df.loc[len(self.df)] = [(str(datetime.datetime.now()-self.start)),self.ang_x, self.ang_y, self.ang_z, self.acc_x, self.acc_y,self.acc_z,0]
                if self.button.double_clicked:
                    print('데이터 수집을 종료합니다.')
                    break

        self.sample_rate = 400
        self.df['bacc_x'] = HPF(self.df['acc_x'], 0.3,self.sample_rate, 3)
        self.df['bacc_y']  = HPF(self.df['acc_y'], 0.3,self.sample_rate, 3)
        self.df['bacc_z'] = HPF(self.df['acc_z'], 0.3,self.sample_rate, 3)
        self.df['gacc_x'] = LPF(self.df['acc_x'], 0.3,self.sample_rate, 3)
        self.df['gacc_y']  = LPF(self.df['acc_y'], 0.3,self.sample_rate, 3)
        self.df['gacc_z'] = LPF(self.df['acc_z'], 0.3,self.sample_rate, 3)

        self.timestamp=[0]
        for i in self.df.index[1:]:
            self.timestamp.append(int(self.df.loc[i]['timestamp'][2:4])*60+int(self.df.loc[i]['timestamp'][5:7])+int(self.df.loc[i]['timestamp'][8:])*0.000001)

        self.df = self.df.drop('timestamp', axis=1) 
        self.df['timestamp_mod'] = self.timestamp
        #self.df.set_index('timestamp_mod', inplace=True)
    

        self.feature_df, self.label = train_preprocess(self.df)
        self.model = train(self.feature_df, self.label)

        return self.model



    def check_botton(self): # 버튼 체크
        if self.button.toggled==True:
            self.led.turn_on()
        elif  self.button.toggled == False:
            self.led.turn_off()
        if self.button.double_clicked:
            self.speaker.tune = 550, 10
        elif self.button.pressed:
            self.speaker.turn_off()


    def get_sensor_data(self, reset): # 센서 데이터 수집
        if reset == 1:
            del[[self.gyrodf,self.envdf]]
            self.gyrodf = pd.DataFrame([[datetime.datetime.now(),0,0,0,0,0,0]], columns = ['timestamp','ang_x','ang_y','ang_z', 'acc_x','acc_y', 'acc_z'])
            self.envdf = pd.DataFrame([[datetime.datetime.now(),0,0,0]], columns = ['timestamp' ,'brightness','humidity','temperature'])
            return self.gyrodf, self.envdf

        self.acc_x = (self.gyro.acceleration_x)/48.5 - self.calib_data[0]
        self.acc_y = (self.gyro.acceleration_y)/48.5- self.calib_data[1]
        self.acc_z = (self.gyro.acceleration_z)/48.5
        self.ang_x = self.gyro.angular_vel_x*3.32- self.calib_data[2]
        self.ang_y = self.gyro.angular_vel_y*3.32- self.calib_data[3]
        self.ang_z = self.gyro.angular_vel_z *3.32- self.calib_data[4]

        self.brightness = self.env.brightness
        self.humidity = self.env.humidity
        self.temperature = self.env.temperature

        self.now = datetime.datetime.now()
        self.gyrodf.loc[len(self.gyrodf)] = [self.now,self.ang_x, self.ang_y,self.ang_z,self.acc_x,self.acc_y,self.acc_z]
        self.envdf.loc[len(self.envdf)] = [self.now,self.brightness, self.humidity,self.temperature]

        return self.gyrodf, self.envdf

    def filtering(self, rawdf, cutoff, order ): # 필터링
        self.sample_rate = int(len(rawdf)/self.interval)
        rawdf['bacc_x'] = HPF(rawdf['acc_x'], cutoff,self.sample_rate, order)
        rawdf['bacc_y']  = HPF(rawdf['acc_y'], cutoff,self.sample_rate, order)
        rawdf['bacc_z'] = HPF(rawdf['acc_z'], cutoff,self.sample_rate, order)
        rawdf['gacc_x'] = LPF(rawdf['acc_x'], cutoff,self.sample_rate, order)
        rawdf['gacc_y']  = LPF(rawdf['acc_y'], cutoff,self.sample_rate, order)
        rawdf['gacc_z'] = LPF(rawdf['acc_z'], cutoff,self.sample_rate, order)
       
        self.timestamp=[0]
        for i in rawdf.index[1:]:
            #self.timestamp.append(int(str(rawdf.loc[i]['timestamp'])[14:16])*60+int(str(rawdf.loc[i]['timestamp'])[17:19])+int(str(rawdf.loc[i]['timestamp'])[20:])*0.000001)
            #self.timestamp.append(rawdf.loc[i]['timestamp'].minute*60+rawdf.loc[i]['timestamp'].second+rawdf.loc[i]['timestamp'].microsecond*0.000001)
            self.timediff = rawdf.loc[i]['timestamp']-rawdf.loc[0]['timestamp']
            self.timestamp.append(self.timediff.seconds+self.timediff.microseconds*0.000001)
        rawdf['timestamp_mod'] = self.timestamp
        rawdf = rawdf.drop('timestamp', axis=1) 
        rawdf['timestamp_mod'] = self.timestamp
        rawdf.to_csv('raw_data.csv')
        #rawdf.set_index('timestamp_mod', inplace=True)


        return rawdf

    def get_position(self, rawdf, model): # 위치 계산
        feature_df = preprocess(rawdf)
        self.df = pred(feature_df, model)
        self.df.to_csv('predic_data.csv')
        self.data = numpy.genfromtxt('predic_data.csv', delimiter=",", skip_header=1)
        self.rawdata =  numpy.genfromtxt('raw_data.csv', delimiter=",", skip_header=1)

        self.sample_rate = int(len(self.rawdata)/self.interval)  # 400 Hz

        self.timestamp = self.rawdata[1:, 13]
        self.gyroscope = self.rawdata[1:, 1:4]
        self.accelerometer = self.rawdata[1:, 4:7] #4:7 - acc, 7:10 - bacc, 10:13 - gacc
        self.movingpredict = self.data[:,13]
        '''
        self.figure, self.axes = pyplot.subplots(nrows=6, sharex=True, gridspec_kw={"height_ratios": [6, 6, 6, 2, 1, 1]})

        self.figure.suptitle("Sensors data, Euler angles, and AHRS internal states")

        self.axes[0].plot(self.timestamp, self.gyroscope[:, 0], "tab:red", label="Gyroscope X")
        self.axes[0].plot(self.timestamp, self.gyroscope[:, 1], "tab:green", label="Gyroscope Y")
        self.axes[0].plot(self.timestamp, self.gyroscope[:, 2], "tab:blue", label="Gyroscope Z")
        self.axes[0].set_ylabel("Degrees/s")
        self.axes[0].grid()
        self.axes[0].legend()

        self.axes[1].plot(self.timestamp, self.accelerometer[:, 0], "tab:red", label="Accelerometer X")
        self.axes[1].plot(self.timestamp, self.accelerometer[:, 1], "tab:green", label="Accelerometer Y")
        self.axes[1].plot(self.timestamp, self.accelerometer[:, 2], "tab:blue", label="Accelerometer Z")
        self.axes[1].set_ylabel("g")
        self.axes[1].grid()
        self.axes[1].legend()
        '''
        #self.accelerometer[:,0] = LPF(self.accelerometer[:,0], 50,self.sample_rate, 3)
        #self.accelerometer[:,1] = LPF(self.accelerometer[:,1], 50,self.sample_rate, 3)
        #self.accelerometer[:,2] = LPF(self.accelerometer[:,2], 50,self.sample_rate, 3)

        # Intantiate AHRS algorithms
        self.offset = imufusion.Offset(self.sample_rate)
        self.ahrs = imufusion.Ahrs()

        self.ahrs.settings = imufusion.Settings(1,  # gain 0.5
                                        10,  # acceleration rejection 10
                                        0,  # magnetic rejection 0
                                        5 * self.sample_rate)  # rejection timeout = 5 seconds 5

        # Process sensor data
        self.delta_time = numpy.diff(self.timestamp, prepend=self.timestamp[0])

        self.euler = numpy.empty((len(self.timestamp), 3))
        self.internal_states = numpy.empty((len(self.timestamp), 3))
        self.acceleration = numpy.empty((len(self.timestamp), 3))

        for index in range(len(self.timestamp)):
            #gyroscope.loc[index] = offset.update(gyroscope.loc[index])

            self.ahrs.update_no_magnetometer(self.gyroscope[index], self.accelerometer[index], self.delta_time[index])

            self.euler[index] = self.ahrs.quaternion.to_euler()

            ahrs_internal_states = self.ahrs.internal_states
            self.internal_states[index] = numpy.array([ahrs_internal_states.acceleration_error,
                                                ahrs_internal_states.accelerometer_ignored,
                                                ahrs_internal_states.acceleration_rejection_timer])

            self.acceleration[index] = 9.81 * self.ahrs.linear_acceleration  # convert g to m/s/s

        self.acceleration[:, 0] = HPF(self.acceleration[:, 0], 0.3,self.sample_rate, 3)
        self.acceleration[:, 1] = HPF(self.acceleration[:, 1], 0.3,self.sample_rate, 3)
        self.acceleration[:, 2] = HPF(self.acceleration[:, 2], 0.3,self.sample_rate, 3)

        '''
        # Plot acceleration
        _, self.axes = pyplot.subplots(nrows=4, sharex=True, gridspec_kw={"height_ratios": [6, 1, 6, 6]})

        self.axes[0].plot(self.timestamp, self.acceleration[:, 0], "tab:red", label="X")
        self.axes[0].plot(self.timestamp, self.acceleration[:, 1], "tab:green", label="Y")
        self.axes[0].plot(self.timestamp, self.acceleration[:, 2], "tab:blue", label="Z")
        self.axes[0].set_title("Acceleration")
        self.axes[0].set_ylabel("m/s/s")
        self.axes[0].grid()
        self.axes[0].legend()
        '''


        # Identify moving periods
        self.is_moving = numpy.empty(len(self.timestamp))
        
        for index in range(len(self.timestamp)):
            self.is_moving[index] = numpy.sqrt(self.acceleration[index].dot(self.acceleration[index])) > 1
        self.margin = int(0.1 * self.sample_rate)  # 100 ms
        
        for index in range(len(self.timestamp) - self.margin):
            self.is_moving[index] = any(self.is_moving[index:(index + self.margin)])  # add leading margin

        for index in range(len(self.timestamp) - 1, self.margin, -1):
            self.is_moving[index] = any(self.is_moving[(index - self.margin):index])  # add trailing margin


        '''
        # Plot moving periods
        self.axes[1].plot(self.timestamp, self.is_moving, "tab:cyan", label="Is moving")
        pyplot.sca(self.axes[1])
        pyplot.yticks([0, 1], ["False", "True"])
        self.axes[1].grid()
        self.axes[1].legend()
        '''
        
        # Calculate velocity (includes integral drift)
        self.velocity = numpy.zeros((len(self.timestamp), 3))

        for index in range(len(self.timestamp)):
            if self.is_moving[index]:  # only integrate if moving
                self.velocity[index] = self.velocity[index - 1] + self.delta_time[index] * self.acceleration[index]

        # Find start and stop indices of each moving period
        self.is_moving_diff = numpy.diff(self.is_moving, append=self.is_moving[-1])


        @dataclass
        class IsMovingPeriod:
            start_index: int = -1
            stop_index: int = -1


        self.is_moving_periods = []
        self.is_moving_period = IsMovingPeriod()

        for index in range(len(self.timestamp)):
            if self.is_moving_period.start_index == -1:
                if self.is_moving_diff[index] == 1:
                    self.is_moving_period.start_index = index

            elif self.is_moving_period.stop_index == -1:
                if self.is_moving_diff[index] == -1:
                    self.is_moving_period.stop_index = index
                    self.is_moving_periods.append(self.is_moving_period)
                    self.is_moving_period = IsMovingPeriod()

        # Remove integral drift from velocity
        self.velocity_drift = numpy.zeros((len(self.timestamp), 3))

        for is_moving_period in self.is_moving_periods:
            self.start_index = is_moving_period.start_index
            self.stop_index = is_moving_period.stop_index

            self.t = [self.timestamp[self.start_index], self.timestamp[self.stop_index]]
            self.x = [self.velocity[self.start_index, 0], self.velocity[self.stop_index, 0]]
            self.y = [self.velocity[self.start_index, 1], self.velocity[self.stop_index, 1]]
            self.z = [self.velocity[self.start_index, 2], self.velocity[self.stop_index, 2]]

            self.t_new = self.timestamp[self.start_index:(self.stop_index + 1)]

            self.velocity_drift[self.start_index:(self.stop_index + 1), 0] = interp1d(self.t, self.x)(self.t_new)
            self.velocity_drift[self.start_index:(self.stop_index + 1), 1] = interp1d(self.t, self.y)(self.t_new)
            self.velocity_drift[self.start_index:(self.stop_index + 1), 2] = interp1d(self.t, self.z)(self.t_new)

        self.velocity = self.velocity - self.velocity_drift
        '''
        # Plot velocity
        self.axes[2].plot(self.timestamp, self.velocity[:, 0], "tab:red", label="X")
        self.axes[2].plot(self.timestamp, self.velocity[:, 1], "tab:green", label="Y")
        self.axes[2].plot(self.timestamp, self.velocity[:, 2], "tab:blue", label="Z")
        self.axes[2].set_title("Velocity")
        self.axes[2].set_ylabel("m/s")
        self.axes[2].grid()
        self.axes[2].legend()
        '''

        # Calculate position
        self.position = numpy.zeros((len(self.timestamp), 3))
        self.position[:, 2] = signal.detrend(self.position[:, 2])



        for index in range(len(self.timestamp)):
            if self.internal_states[index, 2] == 1:
                self.last_index = index
            else : self.last_index = 0
            self.position[index] = self.position[index - 1] + self.delta_time[index] * self.velocity[index]
        '''
        # Plot position
        self.axes[3].plot(self.timestamp, self.position[:, 0], "tab:red", label="X")
        self.axes[3].plot(self.timestamp, self.position[:, 1], "tab:green", label="Y")
        self.axes[3].plot(self.timestamp, self.position[:, 2], "tab:blue", label="Z")
        self.axes[3].set_title("Position")
        self.axes[3].set_xlabel("Seconds")
        self.axes[3].set_ylabel("m")
        self.axes[3].grid()
        self.axes[3].legend()
        pyplot.show()
        '''


        # Print error as distance between start and final positions
        #print("Error: " + "{:.3f}".format(numpy.sqrt(self.position[-1].dot(self.position[-1]))) + " m")
        #print("x : ",position[-1, 0],'y: ',position[-1, 1],"z : ",position[-1, 2])
       
        if abs(self.position[-1, 0])<10 and abs(self.position[-1, 2])<10 and abs(self.position[-1, 2])<10:
            if self.last_index >0:
                return self.position[self.last_index, 0], self.position[self.last_index, 1], self.position[self.last_index, 2], self.euler[self.last_index,0],self.euler[self.last_index,1],self.euler[self.last_index,2]
            return self.position[-1, 0], self.position[-1, 1], self.position[-1, 2], self.euler[-1,0],self.euler[-1,1],self.euler[-1,2]
        else : return 0.0 ,0.0, 0.0, 0.0,0.0,0.0

