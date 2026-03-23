
import aiml
import wikipedia
import csv
import os
import requests
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
from ultralytics import YOLO
import cv2
from collections import Counter
import numpy as np

read_expr = Expression.fromstring

#initialise the knowledge base
kb=[]
data = pandas.read_csv('logical-kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

#checks if the opposite (negative) of each statement exists, if it does it prints an error message and ends the bot
for expr in kb:
    if read_expr(f"-({expr})") in kb:
        print("Knowledge base contains contradictions!  Re-evaluate your knowledge base before rerunning")
        exit()

#creates a kernel object and loads aiml logic from its file
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="waiter_bot_aiml.xml") 

#loads .h5 file containing info like class weights and learnt patterns from CNN
image_classification_model = keras.models.load_model("waiter_classifier.h5")
class_names = ['beer', 'whiskey', 'wine']

#pre-trained model that is used to detect specific objects within images
obj_detection_model = YOLO("yolov8n.pt")

#loads Q&A knweldge base from csv file so the pairs can be used by the chatbot
def load_qa_kb(csv_path: str):
    questions, answers = [], []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            if q and a:
                questions.append(q)
                answers.append(a)
    return questions, answers

class test_similarity_QA:
    #converts words into number values using tfidf
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self.q_matrix = self.vectorizer.fit_transform(self.questions)

    #uses cosign to measure how similar the input is to that of the questions in the Q&A pairs csv
    def answer(self, user_input: str, threshold: float = 0.35):
        if not user_input.strip():
            return None
        vectorise_question = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(vectorise_question, self.q_matrix).ravel()
        best_index = int(similarities.argmax())
        best_score = float(similarities[best_index])
        if best_score >= threshold:
            return self.answers[best_index], best_score, self.questions[best_index]
        return None

#takes an image from the user to make a prediction via the CNN
def predict_drink(image_path: str):
    image = load_img(image_path, target_size=(128, 128))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    prediction = image_classification_model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index])
    return predicted_class, confidence, prediction

#uses yolo model to detect objects in an image given by the user
def detect_drinks(image_path: str):
    model_results = obj_detection_model(image_path, verbose=False)
    result = model_results[0]
    return_image = result.plot()
    output_path = "detected_drinks.jpg"
    cv2.imwrite(output_path, return_image)
    detected_labels = []
    names = result.names

    if result.boxes is not None and len(result.boxes) > 0:
        class_list = result.boxes.cls.tolist()
        confidence_list = result.boxes.conf.tolist()

        for class_id, conf in zip(class_list, confidence_list):
            label = names[int(class_id)]
            if label == "wine glass":
                detected_labels.append("glasses of wine")
            elif label == "bottle":
                detected_labels.append("bottles of wine")

    if len(detected_labels) == 0:
        return "I could not detect any wine glasses or bottles in that image.", output_path, return_image

    counts = Counter(detected_labels)
    wine_object = []

    for label in ["bottles of wine", "glasses of wine"]:
        if label in counts:
            count = counts[label]
            if count == 1:
                wine_object.append(f"1 {label}")
            else:
                wine_object.append(f"{count} {label}")

    final = "I detected: " + ", ".join(wine_object) + "."
    return final, output_path, return_image

API_KEY = "28ccaa0076ec4c91b8cb7b7387c11361"
#uses API to recieve nutritional info via a HTTP GET request
def nutrition_info(drink):

    endpoint = "https://api.spoonacular.com/recipes/guessNutrition"
    params = {
        "title": drink,
        "apiKey": API_KEY
    }
    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        return None
    info = response.json()
    return {
        "calories": info["calories"]["value"]
    }

#returns multiple descriptions of the taste of certain drinks using multi-valued predicates
def describe_drink_tastes(drink_name):
    tastes = []

    for fact in kb:
        fact_string = str(fact)
        if fact_string.startswith("Taste(") and fact_string.endswith(")"):
            cleaned = fact_string[6:-1]  # remove "Taste(" and ")"
            parts = cleaned.split(",")
            if len(parts) == 2:
                kb_drink = parts[0].strip()
                kb_taste = parts[1].strip()
                if kb_drink.lower() == drink_name.lower():
                    tastes.append(kb_taste)

    return tastes

# Load KB once at startup
kb_questions, kb_answers = load_qa_kb("QA.csv")
similarity_QA = test_similarity_QA(kb_questions, kb_answers)

# Welcome user
print("")
print("")
print("")
print("Welcome to this bar waiter chat bot. Please feel free to ask any drink related questions from me!")

# Main loop
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    
    # If AIML returns a command, handle it as you already do
    if answer and answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])

        if cmd == 0:
            print(params[1])
            break

        elif cmd == 1:
            try:
                wiki_summary = wikipedia.summary(params[1], sentences=3, auto_suggest=True)
                print(wiki_summary)

            except:
                # Wikipedia failed → try similarity KB instead
                similarity_result = similarity_QA.answer(userInput, threshold=0.35)

                if similarity_result:
                    kb_answer, score, matched_q = similarity_result
                    print(kb_answer)
                    
                else:
                    print("Sorry, I do not know that. Please be more specific!")

        elif cmd == 60:
            drink = params[1]
            nutrition = nutrition_info(drink)

            if nutrition:
                print(f"A {drink} contains around {nutrition['calories']} calories.")
                
            else:
                print("Sorry, I cannot find the nutritional information requested. Please try again.")

        elif cmd == 71:
            try:
                drink_name = params[1].strip()

                # remove common articles
                for article in ["a ", "an ", "the "]:
                    if drink_name.lower().startswith(article):
                        drink_name = drink_name[len(article):]

                drink_name = drink_name.capitalize()

                tastes = describe_drink_tastes(drink_name)

                if tastes:
                    if len(tastes) == 1:
                        print(f"A {drink_name} tastes {tastes[0].lower()}.")
                    elif len(tastes) == 2:
                        print(f"A {drink_name} tastes {tastes[0].lower()} and {tastes[1].lower()}.")
                    else:
                        taste_list = ", ".join(t.lower() for t in tastes[:-1])
                        print(f"A {drink_name} tastes {taste_list}, and {tastes[-1].lower()}.")
                else:
                    print(f"Sorry, I do not have infomation on what {drink_name} tastes like.")

            except:
                print("Sorry, I could not understand that flavour query. Please try again")

        elif cmd == 99:
            # AIML didn't match -> try similarity KB
            similarity_result = similarity_QA.answer(userInput, threshold=0.35)
            if similarity_result:
                kb_answer, score, matched_q = similarity_result
                print(kb_answer)
                
            else:
                print("I did not get that, please try again.")

        
        elif cmd == 31: # if input pattern is "I know that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            # Build expression: Predicate(Object)
            expr = read_expr(f"{subject}({object})")

            # Contradiction check: would KB entail the negation of this new fact?
            neg_expr = read_expr(f"-({subject}({object}))")

            if ResolutionProver().prove(neg_expr, kb, verbose=False):
                print(f"That would contradict what I already know. I cannot add {expr}.")
            else:
                kb.append(expr)
                print("OK, I will remember that", object, "is", subject)

        elif cmd == 32: # if the input pattern is "check that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=False)
            if answer:
               print("Correct", object, "is", subject)
            else:
                neg_expr = read_expr(f"-({subject}({object}))")
                if ResolutionProver().prove(neg_expr, kb, verbose=False):
                    print('That is incorrect.') 
                else:
                   print("Sorry I don't know, this infomation is not in my knowledge base.")
        
        elif cmd == 40:
            image_path = input("Enter image path: ").strip()

            if os.path.exists(image_path):

                predicted_class, confidence, prediction = predict_drink(image_path)

                print(f"I think this image is {predicted_class}.")

                print(f"Confidence: {confidence:.2f}")
                
            else:
                print("Sorry, I could not classify that image. Please try again.")

        elif cmd == 41:
            image_path = input("Enter image path: ").strip()

            if os.path.exists(image_path):
                final, output_path, return_image = detect_drinks(image_path)
                print(final)
                print(f"Returned image saved as {output_path}")

                cv2.imshow("Drink detections", return_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Sorry, I could not classify that imgage. Please try again.")    

    else:
        # AIML gave a normal answer -> print it
        # If AIML returned empty, also try similarity
        if not answer.strip():
            similarity_result = similarity_QA.answer(userInput, threshold=0.35)
            if similarity_result:
                kb_answer, score, matched_q = similarity_result
                print(kb_answer)
            else:
                print("I did not get that, please try again.")
        else:
            print(answer)