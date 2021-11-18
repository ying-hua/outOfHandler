import cv2 as cv
import json
import math
import numpy as np
import ft2
#v1:1.8,2
#v2:1.8,1.8
#v3:1.8,1.8
#计算两点距离
def getDist(p1,p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
#判断人是否骑在电动车上
def isRide(p):
    pt1=(p['keypoints'][15],p['keypoints'][16])
    pt2=(p['keypoints'][33],p['keypoints'][34])
    pt3=(p['keypoints'][39],p['keypoints'][40])
    pt4=(p['keypoints'][18],p['keypoints'][19])
    pt5=(p['keypoints'][36],p['keypoints'][37])
    pt6=(p['keypoints'][42],p['keypoints'][43])
    dis1=getDist(pt1,pt2)
    dis2=getDist(pt2,pt3)
    dis3=getDist(pt1,pt3)
    dis4=getDist(pt4,pt5)
    dis5=getDist(pt5,pt6)
    dis6=getDist(pt4,pt6)
    cos1=cos2=-1
    if(dis1!=0 and dis2!=0):
        cos1=(dis1*dis1+dis2*dis2-dis3*dis3)/(2*dis1*dis2)
    if(dis4!=0 and dis5!=0):
        cos2=(dis4*dis4+dis5*dis5-dis6*dis6)/(2*dis4*dis5)
    if((cos1>-0.866 or cos2>-0.866) and p['score']>1.8):
        return True
    else:
        return False
def outOfHand(p):
    pt1=(p['keypoints'][15],p['keypoints'][16])
    pt2=(p['keypoints'][21],p['keypoints'][22])
    pt3=(p['keypoints'][27],p['keypoints'][28])
    pt4=(p['keypoints'][18],p['keypoints'][19])
    pt5=(p['keypoints'][24],p['keypoints'][25])
    pt6=(p['keypoints'][30],p['keypoints'][31])
    dis1=getDist(pt1,pt2)
    dis2=getDist(pt2,pt3)
    dis3=getDist(pt1,pt3)
    dis4=getDist(pt4,pt5)
    dis5=getDist(pt5,pt6)
    dis6=getDist(pt4,pt6)
    cos1=cos2=-1
    if(dis1!=0 and dis2!=0):
        cos1=(dis1*dis1+dis2*dis2-dis3*dis3)/(2*dis1*dis2)
    if(dis4!=0 and dis5!=0):
        cos2=(dis4*dis4+dis5*dis5-dis6*dis6)/(2*dis4*dis5)
    if((cos1>0 or cos2>0) and p['score']>1.8):
        return True
    else:
        return False
def putCN(str,img,pos,text_size,color):
    ft = ft2.put_chinese_text('C:\Windows\Fonts\simhei.ttf')
    image = ft.draw_text(img, pos,str, text_size, color)
    return image
#读取视频检测后保存
def save_video():
    #读取视频
    capture=cv.VideoCapture("D:/333/AlphaPose_v2.mp4")
    #capture=cv.VideoCapture(0)
    #新建窗口
    cv.namedWindow("22",cv.WINDOW_NORMAL)
    #设置视频格式
    fourcc=cv.VideoWriter_fourcc(*'mp4v')
    #视频尺寸
    size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
    out=cv.VideoWriter("D:/333/detected_v2.mp4",fourcc,24,size)
    cnt=0   #帧数
    i=0     #人的编号
    with open("D:/333/v2.json",'r') as f:
        jf=json.load(f)
    while(capture.isOpened()):
        det_count=60
        ret,frame=capture.read()
        if ret==False:
            break
        #找出一帧中的所有人
        while(i<len(jf) and jf[i]['image_id']==str(cnt)+".jpg"):
            det_count-=1
            if(isRide(jf[i])==True):
                #画矩形
                pt1=(int(jf[i]['box'][0]),int(jf[i]['box'][1]))
                pt2=(int(jf[i]['box'][0]+jf[i]['box'][2]),int(jf[i]['box'][1]+jf[i]['box'][3]))
                if(outOfHand(jf[i])==True):
                    cv.rectangle(frame,pt1,pt2,(0,0,255),2,8)
                    #打印警告信息
                    #frame=putCN("检测到脱把骑行",frame,(50,50),80,(0,0,255))
                    det_count=60
                else:
                    cv.rectangle(frame,pt1,pt2,(0,255,0),2,8)
            #if(det_count>0):
            #    frame=putCN("检测到脱把骑行",frame,(50,50),80,(0,0,255))
            i+=1
        out.write(frame)
        cv.imshow("22",frame)
        c=cv.waitKey(10)
        if(c==27):
            break
        cnt+=1
    capture.release()
    out.release()
save_video()