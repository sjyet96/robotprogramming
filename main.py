import light
import datetime
import numpy
from multiprocessing import Process, Value, Array
import time
import modi
'''
repeater = modi.MODI()
env = repeater.envs[0]
display = repeater.displays[0]
repeater_cord = 5, 3, 0

def repeater_check():
    env_red = env.red      #빨간색 성분
    env_green = env.green  #초록색 성분 
    env_blue = env.blue    #파란색 성분
    # print(env_red,env_blue,env_green, sep= ' ')
    if (env_green+env_blue)*4< env_red:   # 빨간색이 나머지 두개합친것보다 4배 높을때
        display.text = 'Check!!'
        return True
    else:    
        display.text = 'Need check'
        '''
	 


interval = 10
mylight = light.Light(interval)
model = mylight.get_train_model()
print(model)

start = datetime.datetime.now()
x = 0
y= 0
z=0
while True:
    checktime = start+datetime.timedelta(seconds=interval)
    now = datetime.datetime.now()
    gyrodf , envdf = mylight.get_sensor_data(0)
    mylight.check_botton()
    #if repeater_check():
    #   x, y, z = repeater_cord
     #   continue

    if now >= checktime:
        start = now
        testdf = mylight.filtering(gyrodf, 0.3, 3)
        x_pos, y_pos, z_pos, rol, pitch, yaw = mylight.get_position(testdf, model)  # north west up (west - y, north - z)
        x+= x_pos
        y+= y_pos
        z+= z_pos
        #print(x_pos,y_pos,z_pos, rol, pitch, yaw)
        err = numpy.sqrt(x*x+y*y+z*z)
        print(f'coordinate :({round(x,2)},{round(y,2)},{round(z,2)}), distance : {round(err,2)} ' )
        gyrodf , envdf = mylight.get_sensor_data(1)

