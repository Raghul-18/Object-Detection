import pyttsx3
def textToSpeech(txt):
    engine = pyttsx3.init()
    engine.getProperty('rate')
    engine.setProperty('rate', 200)   
    engine.say(txt)
    engine.runAndWait()


for i in range(1,20,11):
    with open("readme.txt") as fp:
        textToSpeech(fp.readline(i))
