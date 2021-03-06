import json
import cv2
import math
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
    cos1=(dis1*dis1+dis2*dis2-dis3*dis3)/(2*dis1*dis2)
    cos2=(dis4*dis4+dis5*dis5-dis6*dis6)/(2*dis4*dis5)
    if((cos1>-0.866 or cos2>-0.866) and p['score']>1.3):
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
def getDist(p1,p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
src=cv2.imread("D:/333/hij_33.jpeg")
cv2.namedWindow("11",cv2.WINDOW_NORMAL)
with open("D:/333/hij1-54.json",'r') as f:
    jf=json.load(f)
cnt=0
for p in jf:
    if(p['image_id']=='hij_33.jpeg'):
        if(isRide(p)):
            rpt1=(int(p['box'][0]),int(p['box'][1]))
            rpt2=(int(p['box'][0]+p['box'][2]),int(p['box'][1]+p['box'][3]))
            if(outOfHand(p)==True):
                cv2.rectangle(src,rpt1,rpt2,(0,0,255),2,8)
            else:
                cv2.rectangle(src,rpt1,rpt2,(0,255,0),2,8)
            #cnt=cnt+1
            #cv2.putText(src,str(cnt),rpt1,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,8,0)
            #print(cnt,p['score'])
cv2.imshow("11",src)
cv2.waitKey(0)