# import read
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
driver = webdriver.Firefox()
def Average(lst):
    return sum(lst) / len(lst)

CI=30
history = load_model('youtube.h5')
inputsamples=1


def run():
    wait = WebDriverWait(driver, 10)
    motion, name = read.perform(0, inputsamples, CI)
    xmap = interp1d([-2, 2], [0, 99], bounds_error=False, fill_value=(0, 99), kind='linear')
    ymap = interp1d([0, 1.5], [0, 99], bounds_error=False, fill_value=(0, 99), kind='linear')

    BATARR = []
    matrixlstm = np.zeros((inputsamples, CI, 5))
    for alpha in range(1,inputsamples+1):

        arr = []
        for frame in range(0, CI):
            matrix = np.zeros((100, 100, 3))
            xpositions = motion[str(alpha)][frame]['x']
            time.sleep(0.1)
            ypositions = motion[str(alpha)][frame]['y']
            dopplers = motion[str(alpha)][frame]['doppler']
            valrange = motion[str(alpha)][frame]['range']
            peakVal = motion[str(alpha)][frame]['peakVal']
            AvPosx = Average(xpositions)
            AvPosy = Average(ypositions)
            AvDoppler = Average(dopplers)
            AvValrange = Average(valrange)
            AvPeakVal = Average(peakVal)
            xy = zip([int(x) for x in xmap(xpositions)], [int(y) for y in ymap(ypositions)])
            for i, a, b, c in zip(xy, dopplers, valrange, peakVal):
                matrix[i[0]][i[1]] = a, b, c
                matrixlstm[alpha - 1, frame] = AvPosx, AvPosy, AvDoppler, AvValrange, AvPeakVal
            arr.append(matrix)

        motionARR = np.stack(arr, axis=0)
        BATARR.append(motionARR)
    OUTPUTARR = np.stack(BATARR, axis=0)
    OUTPUTARR=np.transpose(OUTPUTARR, (0,2,3,1,4))
    predict_x=history.predict([OUTPUTARR,matrixlstm])
    classes_x = np.argmax(predict_x, axis=1)
    print (classes_x)
    listofword=['youtube','scroll down','scroll up','nothing','pause']
    for i in classes_x:
        if i ==0:

            driver.get("https://www.youtube.com/watch?v=0QNiZfSsPc0")
            play_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@title='Play (k)']")))
            play_btn.click()

        if i==1:
            html = driver.find_element_by_tag_name('html')
            html.send_keys(Keys.PAGE_DOWN)
        if i==2:
            html = driver.find_element_by_tag_name('html')
            html.send_keys(Keys.HOME)
        if i==4:
            mute_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Mute (m)']")))
            mute_btn.click()
        return (listofword[i])


    # history.history.keys()
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
while True:
    print(run())
    time.sleep(3)