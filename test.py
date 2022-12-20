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

timediff=[0]
acc_x=[0]
acc_y=[0]
acc_z=[0]
vel_x=[0]
vel_y=[0]
vel_z=[0]
dis_x=[0]
dis_y=[0]
dis_z=[0]


def BPF(series, cutoff_low, cutoff_high,fs, order=1):
    '''
    series : 데이터
    low : 최저 구간(0< low < 1)
    order : 필터 계수, 높을수록 민감
    '''
    
    nyq = 0.5*fs
    cutoff_low = cutoff_low/nyq
    cutoff_high = cutoff_high/nyq
    
    b, a = butter(
                  N = order,
                  Wn = [cutoff_low,cutoff_high],
                  btype = 'band',
                  )
    lpf_series = signal.filtfilt(b, a, series)
    
    return lpf_series



def LPF(series, cutoff,fs, order=1):
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

def HPF(series, cutoff,fs, order=1):
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

def calib_gyro():
    acc_x=[0]
    acc_y=[0]
    acc_z=[0]
    
   
    i=0
    df = pd.DataFrame([[datetime.datetime.now(),0,0,0,0,0,0]], columns = ['timestamp','acc_x','acc_y','acc_z', 'ang_x','ang_y', 'ang_z'])
    print("calibrating...")
    while True:
        acc_x = (gyro.acceleration_x)/48.5
        acc_y = (gyro.acceleration_y)/48.5
        acc_z = (gyro.acceleration_z)/48.5
        ang_x = gyro.angular_vel_x*3.32
        ang_y = gyro.angular_vel_y*3.32
        ang_z = gyro.angular_vel_z *3.32
        #rol = gyro.roll
        #pitch = gyro.pitch
        #yaw = gyro.yaw
        df.loc[len(df)] = [datetime.datetime.now(),acc_x, acc_y,acc_z,ang_x,ang_y,ang_z]
        i +=1
        if i>2000:
            print("calibration done")
            break

    calib_acc_x = df['acc_x'].mean()
    calib_acc_y = df['acc_y'].mean()
    calib_acc_z = df['acc_z'].mean()
    #calib_rol = df['rol'].mean()
    #calib_pitch = df['pitch'].mean()
    #calib_yaw = df['yaw'].mean()
    calib_ang_x = df['ang_x'].mean()
    calib_ang_y = df['ang_y'].mean()
    calib_ang_z = df['ang_z'].mean()
    #print(calib_acc_x,calib_acc_y,calib_acc_z, calib_rol,calib_pitch,calib_yaw)
    result = [calib_acc_x,calib_acc_y,calib_acc_z, calib_ang_x,calib_ang_y,calib_ang_z]
    return result

bundle = modi.MODI(
    #conn_type='ble',network_uuid="611DA680"
    )

gyro=bundle.gyros[0]
network = bundle.networks[0]

i=0
calib_data= calib_gyro()
#time.sleep(1)
df = pd.DataFrame([[0,0,0,0,0,0,0]], columns = ['timestamp','ang_x', 'ang_y','ang_z','acc_x','acc_y','acc_z'])
start = datetime.datetime.now()
end = start+datetime.timedelta(seconds=15)
now = datetime.datetime.now()
while True:
    time.sleep(0.01)
    now = datetime.datetime.now()
    acc_x=0
    acc_y=0
    acc_z=0
    rol=0
    pitch=0
    yaw=0
    ## 센서값 측정

    acc_x = ((gyro.acceleration_x)/48.5)-calib_data[0]
    acc_y = ((gyro.acceleration_y)/48.5)-calib_data[1]
    acc_z = ((gyro.acceleration_z)/48.5)-calib_data[1]
   
    ang_x = gyro.angular_vel_x*3.32-calib_data[3]
    ang_y = gyro.angular_vel_y*3.32-calib_data[4]
    ang_z = gyro.angular_vel_z *3.32-calib_data[5]
    
    print(acc_x,acc_y,acc_z)

    ## 데이터 프레임에 센서값 저장 ##
    df.loc[len(df)] = [(str(datetime.datetime.now()-start)[5:]),ang_x, ang_y, ang_z, acc_x, acc_y,acc_z]
    if now>=end:
        break

df.set_index('timestamp', inplace=True)
#lowpass = signal.firwin(3,cutoff = 0.3, fs = 1300, pass_zero='highpass')
#df['acc_x'] = signal.lfilter(lowpass,[1,0],df['acc_x'])
#df['acc_y'] = signal.lfilter(lowpass,[1,0],df['acc_y'])
#df['acc_z'] = signal.lfilter(lowpass,[1,0],df['acc_z'])


#df['acc_x'] = HPF(df['acc_x'], 0.3,1300, 3)
#df['acc_y'] = HPF(df['acc_y'], 0.3,1300, 3)
#df['acc_z'] = HPF(df['acc_z'], 0.3,1300, 3)
#df['ang_x'] = HPF(df['ang_x'], 0.3, 3)
#df['ang_y'] = HPF(df['ang_y'], 0.3, 3)
#df['ang_z'] = HPF(df['ang_z'], 0.3, 3)

#df['acc_z'] = LPF(df['acc_z'], 0.1, 1)
df.to_csv('/Users/songjiyu/python codes/robotProgramming/프로젝트/df.csv')
print(df)
# Import sensor data ("short_walk.csv" or "long_walk.csv")
#data = numpy.genfromtxt("/Users/songjiyu/python codes/robotProgramming/프로젝트/Gait-Tracking-main/long_walk.csv", delimiter=",", skip_header=1)

data = numpy.genfromtxt("df.csv", delimiter=",", skip_header=1)

sample_rate = int(len(df)/15)  # 400 Hz

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]

# Plot sensor data
figure, axes = pyplot.subplots(nrows=6, sharex=True, gridspec_kw={"height_ratios": [6, 6, 6, 2, 1, 1]})

figure.suptitle("Sensors data, Euler angles, and AHRS internal states")

axes[0].plot(timestamp, gyroscope[:, 0], "tab:red", label="Gyroscope X")
axes[0].plot(timestamp, gyroscope[:, 1], "tab:green", label="Gyroscope Y")
axes[0].plot(timestamp, gyroscope[:, 2], "tab:blue", label="Gyroscope Z")
axes[0].set_ylabel("Degrees/s")
axes[0].grid()
axes[0].legend()

axes[1].plot(timestamp, accelerometer[:, 0], "tab:red", label="Accelerometer X")
axes[1].plot(timestamp, accelerometer[:, 1], "tab:green", label="Accelerometer Y")
axes[1].plot(timestamp, accelerometer[:, 2], "tab:blue", label="Accelerometer Z")
axes[1].set_ylabel("g")
axes[1].grid()
axes[1].legend()

# Intantiate AHRS algorithms
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(0.5,  # gain 0.5
                                   10,  # acceleration rejection 10
                                   0,  # magnetic rejection 0
                                   5 * sample_rate)  # rejection timeout = 5 seconds 5

# Process sensor data
delta_time = numpy.diff(timestamp, prepend=timestamp[0])

euler = numpy.empty((len(timestamp), 3))
internal_states = numpy.empty((len(timestamp), 3))
acceleration = numpy.empty((len(timestamp), 3))

for index in range(len(timestamp)):
    #gyroscope.loc[index] = offset.update(gyroscope.loc[index])

    ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])

    euler[index] = ahrs.quaternion.to_euler()

    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = numpy.array([ahrs_internal_states.acceleration_error,
                                          ahrs_internal_states.accelerometer_ignored,
                                          ahrs_internal_states.acceleration_rejection_timer])

    acceleration[index] = 9.81 * ahrs.earth_acceleration  # convert g to m/s/s
acceleration[:, 0] = HPF(acceleration[:, 0], 0.3, sample_rate, 3)
acceleration[:, 1] = HPF(acceleration[:, 1], 0.3,sample_rate, 3)
acceleration[:, 2] = HPF(acceleration[:, 2], 0.3,sample_rate, 3)

# Plot Euler angles
axes[2].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
axes[2].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
axes[2].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
axes[2].set_ylabel("Degrees")
axes[2].grid()
axes[2].legend()

# Plot internal states
axes[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
axes[3].set_ylabel("Degrees")
axes[3].grid()
axes[3].legend()

axes[4].plot(timestamp, internal_states[:, 1], "tab:cyan", label="Accelerometer ignored")
pyplot.sca(axes[4])
pyplot.yticks([0, 1], ["False", "True"])
axes[4].grid()
axes[4].legend()

axes[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration rejection timer")
axes[5].set_xlabel("Seconds")
axes[5].grid()
axes[5].legend()

# Plot acceleration
_, axes = pyplot.subplots(nrows=4, sharex=True, gridspec_kw={"height_ratios": [6, 1, 6, 6]})

axes[0].plot(timestamp, acceleration[:, 0], "tab:red", label="X")
axes[0].plot(timestamp, acceleration[:, 1], "tab:green", label="Y")
axes[0].plot(timestamp, acceleration[:, 2], "tab:blue", label="Z")
axes[0].set_title("Acceleration")
axes[0].set_ylabel("m/s/s")
axes[0].grid()
axes[0].legend()

# Identify moving periods
is_moving = numpy.empty(len(timestamp))

for index in range(len(timestamp)):
    is_moving[index] = numpy.sqrt(acceleration[index].dot(acceleration[index])) > 3  # threshold = 3 m/s/s

margin = int(0.1 * sample_rate)  # 100 ms

for index in range(len(timestamp) - margin):
    is_moving[index] = any(is_moving[index:(index + margin)])  # add leading margin

for index in range(len(timestamp) - 1, margin, -1):
    is_moving[index] = any(is_moving[(index - margin):index])  # add trailing margin

# Plot moving periods
axes[1].plot(timestamp, is_moving, "tab:cyan", label="Is moving")
pyplot.sca(axes[1])
pyplot.yticks([0, 1], ["False", "True"])
axes[1].grid()
axes[1].legend()

# Calculate velocity (includes integral drift)
velocity = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    if is_moving[index]:  # only integrate if moving
        velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

# Find start and stop indices of each moving period
is_moving_diff = numpy.diff(is_moving, append=is_moving[-1])


@dataclass
class IsMovingPeriod:
    start_index: int = -1
    stop_index: int = -1


is_moving_periods = []
is_moving_period = IsMovingPeriod()

for index in range(len(timestamp)):
    if is_moving_period.start_index == -1:
        if is_moving_diff[index] == 1:
            is_moving_period.start_index = index

    elif is_moving_period.stop_index == -1:
        if is_moving_diff[index] == -1:
            is_moving_period.stop_index = index
            is_moving_periods.append(is_moving_period)
            is_moving_period = IsMovingPeriod()

# Remove integral drift from velocity
velocity_drift = numpy.zeros((len(timestamp), 3))

for is_moving_period in is_moving_periods:
    start_index = is_moving_period.start_index
    stop_index = is_moving_period.stop_index

    t = [timestamp[start_index], timestamp[stop_index]]
    x = [velocity[start_index, 0], velocity[stop_index, 0]]
    y = [velocity[start_index, 1], velocity[stop_index, 1]]
    z = [velocity[start_index, 2], velocity[stop_index, 2]]

    t_new = timestamp[start_index:(stop_index + 1)]

    velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x)(t_new)
    velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y)(t_new)
    velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z)(t_new)

velocity = velocity - velocity_drift

#velocity[:, 0] = LPF(velocity[:, 0], 0.3,sample_rate, 3)
#velocity[:, 1] = LPF(velocity[:, 1], 0.3,sample_rate, 3)
#velocity[:, 2] = LPF(velocity[:, 2], 0.5,sample_rate, 3)

# Plot velocity
axes[2].plot(timestamp, velocity[:, 0], "tab:red", label="X")
axes[2].plot(timestamp, velocity[:, 1], "tab:green", label="Y")
axes[2].plot(timestamp, velocity[:, 2], "tab:blue", label="Z")
axes[2].set_title("Velocity")
axes[2].set_ylabel("m/s")
axes[2].grid()
axes[2].legend()

# Calculate position
position = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    position[index] = position[index - 1] + delta_time[index] * velocity[index]

position[:, 2] = HPF(position[:, 2], 0.3,sample_rate, 3)
position[:, 2] = HPF(position[:, 2], 0.3,sample_rate, 3)
position[:, 2] = HPF(position[:, 2], 0.3,sample_rate, 3)

position[:,2] = signal.detrend(position[:,2])
# Plot position
axes[3].plot(timestamp, position[:, 0], "tab:red", label="X")
axes[3].plot(timestamp, position[:, 1], "tab:green", label="Y")
axes[3].plot(timestamp, position[:, 2], "tab:blue", label="Z")
axes[3].set_title("Position")
axes[3].set_xlabel("Seconds")
axes[3].set_ylabel("m")
axes[3].grid()
axes[3].legend()

# Print error as distance between start and final positions
print("Error: " + "{:.3f}".format(numpy.sqrt(position[-1].dot(position[-1]))) + " m")
print("x : ",position[-1, 0],'y: ',position[-1, 1],"z : ",position[-1, 2])


for index in range(len(timestamp)):
    if internal_states[index, 2] == 1:
        print(position[index], numpy.sqrt(position[index].dot(position[index])))


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.xlim(min(position[:, 0]), max(position[:, 0]))
plt.ylim(min(position[:, 1]), max(position[:, 1]))


ax.scatter(position[:, 0],position[:, 1],position[:, 2])
plt.show()


# Create 3D animation (takes a long time, set to False to skip)
if False:
    figure = pyplot.figure(figsize=(10, 10))

    axes = pyplot.axes(projection="3d")
    axes.set_xlabel("m")
    axes.set_ylabel("m")
    axes.set_zlabel("m")

    x = []
    y = []
    z = []

    scatter = axes.scatter(x, y, z)

    fps = 30
    samples_per_frame = int(sample_rate / fps)

    def update(frame):
        index = frame * samples_per_frame

        axes.set_title("{:.3f}".format(timestamp[index]) + " s")

        x.append(position[index, 0])
        y.append(position[index, 1])
        z.append(position[index, 2])

        scatter._offsets3d = (x, y, z)

        if (min(x) != max(x)) and (min(y) != max(y)) and (min(z) != max(z)):
            axes.set_xlim3d(min(x), max(x))
            axes.set_ylim3d(min(y), max(y))
            axes.set_zlim3d(min(z), max(z))

            axes.set_box_aspect((numpy.ptp(x), numpy.ptp(y), numpy.ptp(z)))

        return scatter

    anim = animation.FuncAnimation(figure, update,
                                   frames=int(len(timestamp) / samples_per_frame),
                                   interval=1000 / fps,
                                   repeat=False)

    anim.save("animation.gif", writer=animation.PillowWriter(fps))

pyplot.show()