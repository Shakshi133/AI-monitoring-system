import speech_recognition as sr
import os
from gtts import gTTS

recognizer=sr.Recognizer()
print("Do you want to record audio or want to add audio : ")
choice=int(input("for recording press 1 and To add audio press 0"))
if choice==0:
    audiofile=input("enter your path for audio file: ")
    with sr.AudioFile(audiofile) as source:
        try:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            print("Transcription: ", text)
            
        except sr.UnknownValueError:
            print("Could not understand audio")
            
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            
   
    
if choice==1:
    
    with sr.Microphone() as source:
        print("Start Recording: ")
        recognizer.adjust_for_ambient_noise(source,duration=1)
        try:
            audio=recognizer.listen(source,timeout=5)
            text = recognizer.recognize_google(audio, language='en-US')
            print("You said: " + text)
       
    
        except sr.UnknownValueError:
            print("Sorry, I could not understand the speech.")
    
        except sr.RequestError as e:
             print(f"Could not request results from Google Speech Recognition service; {e}")
    
        except Exception as e:
             print(f"An error occurred: {e}")

if text:
            t=int(input("If you want to convert it into audio, press 1"))
            if(t==1):
                tts=gTTS(text=text,lang='en',slow=False)
                tts.save("output.mp3")
            else:
                print("We have successfully converted your audio into text: ")
    

            # Play the speech (works on both Windows and Linux/Mac)
            if os.name == 'nt':  # Windows
                os.system("start output.mp3")
            else:  # Linux or macOS
                os.system("mpg321 output.mp3")  # Ensure mpg321 is installed or use another player
else:
     print("No speech detected.")

