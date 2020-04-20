def recognition():
    ###   數字辨識   ###
    
    #from datetime import datetime # 計算runtime
    import cv2 # 做圖片處理
    import numpy as np # 產生陣列
    import PIL # 做圖片處理
    import pyautogui # 截圖
        
    #StartTime = datetime.now() # 開始時間
    
    loop=0
    
    # 圖片分割線
    separatedline = 44 # 距離左邊多少 
    
    
    
    #########################  訓練模型  ################################ 
    
    samples = np.loadtxt('generalsamples.txt',np.float32) # 載入先前儲存來訓練的資料
    responses = np.loadtxt('generalresponses.txt',np.float32) # 載入使用者按下的數字的資料
    responses = responses.reshape((responses.size,1)) # 將使用者按下的數字的資料個別用一個陣列儲存
    model = cv2.ml.KNearest_create() # 創建模型
    model.train(samples,cv2.ml.ROW_SAMPLE,responses) # 訓練模型
    
    
    ########################## 處理 + 辨識圖片  ################################
      
    
    ###   讀取 + 圖片處理   ###
    
    im = pyautogui.screenshot(region =(61,360,264,40) )
    im.save("images/screenshot%d.png"%loop)
    im = cv2.imread("images/screenshot%d.png"%loop) # 讀取要辨識圖片
    
    
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #將要辨識的圖片轉成灰階
    imh,imw,imc = im.shape # 取得圖片的h,w,c
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #將要辨識的圖片轉成灰階
    a, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV) # 將要辨識的圖片提高對比度
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # 找輪廓
    
    
    
    ###   取得所有數字的矩形   ###
    
    bounding = []
    lastval2 = 0
    
    for cnt in reversed(contours): # 因為輪廓主要是從右邊到左邊偵測(x從大到小)，所以用reversed將輪廓反轉過來變成(x從小到大)
        [x,y,w,h] = cv2.boundingRect(cnt) # 取得轉成矩形的輪廓的x,y,w,h
    
    
        if h > 7:
            if x < lastval2:
                break
            
            elif w > 15 and x > lastval2 : # 當辨識到較寬的數字時將數字分割(3位數)
                bounding.append([x,y,int(w/2)-5 ,h]) # 加入左邊的數字的x,y,w,h
                bounding.append([x+int(w/2)-2,y,int(w/2)-5,h]) # 加入中間的數字的x,y,w,h
                bounding.append([x+int(w/2)+5,y,int(w/2)+1,h]) # 加入右邊的數字x,y,w,h
            
            elif w > 8 and x > lastval2 : # 當辨識到較寬的數字時將數字分割(2位數)

                bounding.append([x,y,int(w/2)-1,h]) # 加入左邊的數字的x,y,w,h
                bounding.append([x+int(w/2)+1,y,int(w/2)+1,h]) # 加入右邊的數字x,y,w,h
            
            elif x > lastval2:
                bounding.append([x,y,w,h]) # 加入數字的x,y,w,h
            
            
        lastval2 = x # 將上一個矩形的x保存下來
        
        
    
    ###   判斷數字與分割線的位置   ###
    
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    s6 = 0
    
    for u in bounding:
        [x,y,w,h] = u
        
        if 0 < x + (w/2) <= separatedline:
            s1+=1            
        elif separatedline+1 < x + (w/2) <= separatedline*2+1:
            s2+=1
        elif separatedline*2+2 < x + (w/2) <= separatedline*3+2:
            s3+=1
        elif separatedline*3+3 < x + (w/2) <= separatedline*4+3:
            s4+=1
        elif separatedline*4+4 < x + (w/2) <= separatedline*5+4:
            s5+=1
        elif separatedline*5+4 < x :
            s6+=1                
            
    space = [s1,s1+s2,s1+s2+s3,s1+s2+s3+s4,s1+s2+s3+s4+s5,s1+s2+s3+s4+s5+s6] # 從左到右數字出現數累加

    

    
    ###   將上面的矩陣當作分割圖片的依據，並將分割的圖片貼到新圖片上   ###
    
    loop2 = 1 # 給圖片命名用的
    count = 0 # 第一次不做
    count2 = 1 # 用來計算新圖片x的位置
    lastxw = 0 # 上一個矩形的x+w 
    lastlx = 0 # 分割圖片的起點
    lastsep = 0 # 貼在新圖片的x位置 
    newimg = PIL.Image.new("L",(imw+160,imh),"white") # 產生新的空白圖片
    
    for cnt in bounding: 
        [x,y,w,h] = cnt # 取得矩陣的x,y,w,h
        
            
        if (x < lastxw + 7) and (count != 0): # 限制數字間相距的距離
            
            Range = gray[0:0+imh ,lastlx :lastxw ] # 圖片的分割範圍
            cv2.imwrite("separated/ss%d.png" %loop2,Range) # 將分割儲存圖片
            im = PIL.Image.open("separated/ss%d.png" %loop2 ) # 開啟分割圖片
            newimg.paste(im, (lastsep+20, 0)) # 把圖片貼到新的空白圖片上
            lastsep = lastxw + 10*count2 # 貼在新圖片的x位置           
            lastlx = x -1 # 分割圖片的起點
            loop2 += 1 
            count2 += 1
            
        lastxw = x + w +1  # 分割圖片的結尾
        count = 1 # 第二次開始做
        
        ss = gray[0:0+imh,lastlx:lastxw] # 圖片的分割範圍
        cv2.imwrite("separated/ss%d.png" %loop2,ss) # 儲存圖片分割的圖片
        im = PIL.Image.open("separated/ss%d.png" %loop2 ) # 用PIL開啟分割的圖片
        newimg.paste(im, (lastsep+20, 0) ) # 把圖片貼到新的空白圖片上 
            
    newimg.save("separated/separated.png") # 將新圖片用PIL儲存
    im = cv2.imread("separated/separated.png") # 讀取要辨識圖片
    
    
    
    ###   將分割完的圖片做處理   ###
    
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) # 將要辨識的圖片轉成灰階
    a, im = cv2.threshold(im,120,255,cv2.THRESH_BINARY) # 將要辨識的圖片提高對比度
    im = cv2.adaptiveThreshold(im,255,1,1,11,2) # 將要辨識的圖片做自適二極化，圖片將變成黑白
    cv2.imwrite("thresh/screenshot%d.png"%loop,im) # 儲存圖片
    contours,hierarchy = cv2.findContours(im,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # 找輪廓
    im = PIL.Image.open("thresh/screenshot%d.png" %loop) # 用PIL開啟圖片
    
    
    
    ###   分割圖片的每個數字 + 辨識   ###    
    
    lastxw2 = 0 # 上一個矩形的w+x
    loop3 = 1 # 給圖片命名用的
    string2 = "" # 儲存數字的新字串
    
    for cnt in reversed(contours):
        [x,y,w,h] = cv2.boundingRect(cnt)
           
        if  x > lastxw2: # 限制高度以及避免出現重疊的座標
            lastxw2 = x+w # 上一個矩形的w+x
            num = im.crop((x,y,x+w,y+h)) # 用PIL分割圖片
            num.save("numbers/num%d.png" %loop3) # 用PIL儲存圖片
            num = cv2.imread("numbers/num%d.png" %loop3,cv2.IMREAD_GRAYSCALE) # 用CV2讀取圖片          
            renum = cv2.resize(num,(10,10)) # 用CV2將圖片重設為10X10
            renum = renum.reshape((1,100)) # 將10x10圖片的像素存儲在空間為100的矩陣
            renum = np.float32(renum)
            retval, results, neigh_resp, dists = model.findNearest(renum, k = 1) # 找最近的數字
            string = str(int( (results[0][0]) ))
            string2 += string # 把辨識的數字加到新字串中當中
            loop3 += 1
            
    retry = False
    if s1==0 or s2==0 or s3==0 or s4==0 or s6 == 0 or s1>= 4 or s2>= 4 or s3>= 4 or s4>= 4 or s6 >= 4 or s1+s2+s3+s4+s5+s6 !=len(string2):
        print (s1,s2,s3,s4,s5,s6,len(string2))
        retry = True
        return retry
    
    else:
        ###   分割數字串   ###
        count3 = 0
        first1 = True
        first2 = True
        first3 = True
        first4 = True
        first5 = True
        sepstring = ""
        
        
      
        # 保存紀錄
        savetxt = open('record.txt',"a",encoding="utf-8") # 打開用來儲存當前辨識數字的txt檔案
    
            
        for k in string2:
            
            if count3 < space[0] : 
                sepstring += k
                count3+=1
                
            elif space[0] <= count3 < space[1] :
                if first1 == True:
                    savetxt.write("x軸:" + ((3-len(sepstring))*"0"+sepstring).rjust(4)+"\n" )
                    #print ("x軸:" + ((3-len(sepstring))*"0"+sepstring).rjust(4) )
                    first1 = False
                    sepstring = ""
                sepstring += k
                count3+=1
                
            elif space[1] <= count3 < space[2] :
                if first2 == True:                
                    savetxt.write("y軸:" + ((3-len(sepstring))*"0"+sepstring).rjust(4)+"\n" )
                    #print("y軸:" + ((3-len(sepstring))*"0"+sepstring).rjust(4) )
                    first2 = False
                    sepstring = ""
                sepstring += k
                count3+=1
                
            elif space[2] <= count3 < space[3] :
                if first3 == True:
                    savetxt.write("z軸:" + ((3-len(sepstring))*"0"+sepstring).rjust(4)+"\n" )
                    #print("z軸:" + ((3-len(sepstring))*"0"+sepstring).rjust(4) )
                    first3 = False
                    sepstring = ""
                sepstring += k
                count3+=1
                
            elif space[3] <= count3 < space[4] :
                if first4 == True:
                    savetxt.write("A站:" + ((3-len(sepstring))*"0"+sepstring).rjust(4)+"\n" )
                    #print("A站:" + ((3-len(sepstring))*"0"+sepstring).rjust(4) )
                    first4 = False
                    sepstring = ""
                sepstring += k
                count3+=1
                
            elif space[4] <= count3 < space[5] :
                if first5 == True:
                    savetxt.write("B站:" + ((3-len(sepstring))*"0"+sepstring).rjust(4)+"\n" )
                    #print("B站:" + ((3-len(sepstring))*"0"+sepstring).rjust(4) )
                    first5 = False
                    sepstring = ""
                sepstring += k
                count3+=1
        
        savetxt.write("C站:" + ((3-len(sepstring))*"0"+sepstring).rjust(4)+"\n" )
        #print("C站:" + ((3-len(sepstring))*"0"+sepstring).rjust(4) )
        
        #print (datetime.now() - StartTime) # 顯示總時間
        
        savetxt.close() # 關閉txt
        #print (s1,s2,s3,s4,s5,s6)
        return retry
