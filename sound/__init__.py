import pyttsx3
import time

# import win32com.client
# #微软这个服务器
# speaker = win32com.client.Dispatch("SAPI.SpVoice")
# speaker.Speak("123")
'''
#更改声音(音色处理)
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
print(len(voices))

for voice in voices:
    engine.setProperty('voice', voice.id)
    print(voice.id)
    engine.say('안녕하세요 ')
    engine.runAndWait()
'''
str = "강우씨, 찾아주셔서 감사합니다."
engine = pyttsx3.init()
engine.setProperty('voice', engine.getProperty('voices')[2].id)
num = 0
while num < 3:
    engine.say(str)
    engine.runAndWait()
    num += 1
    time.sleep(1)



