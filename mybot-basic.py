#!/usr/bin/env python3

# A basic chatbot design --- a starting point for developing your own chatbot

#######################################################
#  Initialise AIML agent
import aiml
import wikipedia
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
read_expr = Expression.fromstring

#  Initialise Knowledgebase. 
import pandas
kb=[]
data = pandas.read_csv('logical-kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

for expr in kb:
    if read_expr(f"-({expr})") in kb:
        print("ERROR: Knowledge base contains contradictions!  Re-evaluate your knowledge base before rerunning")
        exit()

print("Logical knowledge base loaded successfully. No contradictions found.")

# above does not catch inferred contradictions.

# Create a Kernel object. 
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

image_model = keras.models.load_model("alcohol_classifier.h5")
class_names = ['beer', 'whiskey', 'wine']

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

class SimilarityQA:
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self.q_matrix = self.vectorizer.fit_transform(self.questions)

    def answer(self, user_query: str, threshold: float = 0.35):
        if not user_query.strip():
            return None
        q_vec = self.vectorizer.transform([user_query])
        sims = cosine_similarity(q_vec, self.q_matrix).ravel()
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        if best_score >= threshold:
            return self.answers[best_idx], best_score, self.questions[best_idx]
        return None

def predict_drink(image_path: str):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    
    img_array = np.expand_dims(img_array, axis=0)

    prediction = image_model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index])

    return predicted_class, confidence, prediction

# Load KB once at startup
kb_questions, kb_answers = load_qa_kb("cocktail_QA_high_paraphrase.csv")
sim_qa = SimilarityQA(kb_questions, kb_answers)

# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!")

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
    #post-process the answer for commands

    # If AIML returns a command, handle it as you already do
    if answer and answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])

        if cmd == 0:
            print(params[1])
            break

        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=True)
                print(wSummary)

            except:
                # Wikipedia failed → try similarity KB instead
                sim_result = sim_qa.answer(userInput, threshold=0.35)

                if sim_result:
                    kb_answer, score, matched_q = sim_result
                    print(kb_answer)
                    # Optional debug:
                    # print(f"(matched: '{matched_q}' score={score:.2f})")
                else:
                    print("Sorry, I do not know that. Be more specific!")

        elif cmd == 99:
            # AIML didn't match -> try similarity KB
            sim_result = sim_qa.answer(userInput, threshold=0.35)
            if sim_result:
                kb_answer, score, matched_q = sim_result
                print(kb_answer)
                # Optional debug (remove for final submission):
                # print(f"(matched: '{matched_q}' score={score:.2f})")
            else:
                print("I did not get that, please try again.")

        # Here are the processing of the new logical component:
        elif cmd == 31: # if input pattern is "I know that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            # Build expression: Predicate(Object)
            expr = read_expr(f"{subject}({object})")

            # Contradiction check: would KB entail the negation of this new fact?
            neg_expr = read_expr(f"-({subject}({object}))")

            if ResolutionProver().prove(neg_expr, kb, verbose=False):
                print(f"Error: That would contradict what I already know. I cannot add {expr}.")
            else:
                kb.append(expr)
                print("OK, I will remember that", object, "is", subject)

        elif cmd == 32: # if the input pattern is "check that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
               print('Correct.')
            else:
                neg_expr = read_expr(f"-({subject}({object}))")
                if ResolutionProver().prove(neg_expr, kb, verbose=False):
                    print('Incorrect.') 
                else:
                   print("Sorry I don't know.")
        
        elif cmd == 40:
            image_path = input("Enter image path: ").strip()

            if os.path.exists(image_path):

                predicted_class, confidence, prediction = predict_drink(image_path)

                if confidence < 0.6:
                    print(f"I'm not overly confident, but this image might be {predicted_class}.")
                else:
                    print(f"I think this image is {predicted_class}.")

                print(f"Confidence: {confidence:.2f}")
                print("Probabilities:")

                for i, name in enumerate(class_names):
                    print(f"{name}: {prediction[i]:.4f}")

            else:
                print("Sorry, I could not load or classify that image.")
    else:
        # AIML gave a normal answer -> print it
        # If AIML returned empty, also try similarity
        if not answer.strip():
            sim_result = sim_qa.answer(userInput, threshold=0.35)
            if sim_result:
                kb_answer, score, matched_q = sim_result
                print(kb_answer)
            else:
                print("I did not get that, please try again.")
        else:
            print(answer)