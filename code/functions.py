import numpy as np

a = [2,4,6]
b = [0,0,0]
c = [-1,3,2]
def calculate_angle(a,b,c):  #function only works in two dimensions as written
    a = np.array(a)  #first landmark
    b = np.array(b)  #second landmark
    c = np.array(c)  #third landmark

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

#attempt at 3d rendering
'''def calculate_angle(a,b,c):
    a = np.array(a)  # first landmark
    b = np.array(b)  # second landmark
    c = np.array(c)  # third landmark
    
    ab = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    cb = c[0] - b[0], c[1] - b[1], c[2] - b[2]
    
    dot = 0
    for i in range(3):
        dot += ab[i] * cb[i]
    
    abNorm = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    cbNorm = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2)
    
    angle = np.degrees(np.arccos(dot / (abNorm * cbNorm)))
    
    if angle > 180.0:
        angle = 360 - angle
    return angle'''

calculate_angle(a, b, c)