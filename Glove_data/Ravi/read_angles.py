import serial
import struct
import time
import numpy as np
import matplotlib.pyplot as plt


ser = serial.Serial('COM3', 9600, timeout=0)

messageLength = 7
data = np.zeros(messageLength)
data2 = [data]

messageCount = 0;
startTime = time.time()
while time.time() - startTime < 5:
    # print(time.time() - startTime)
    try:
        val = ser.read(1)
        if (val == b'+'):
            for i in range(0, messageLength):
                val2 = b''
                readCount = 0
                while readCount < 8:
                    val = ser.read(1)
                    readCount += len(val)
                    val2 += val
                val2 = struct.unpack('d', val2)
                data[i] = val2[0]
            data2 = np.append(data2, [data], axis = 0)
        messageCount += 1;
        # time.sleep(.01)
    except serial.SerialTimeoutException:
        print('Data could not be read')
        time.sleep(.01)

    print(data)

data2 = data2[5:, :]
data2[:, 0] = data2[:, 0] - data2[0, 0]
ans = input('name this set\n')

if ans != 'n':
    np.save('results/'+ans, data2)
