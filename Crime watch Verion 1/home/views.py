from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import logout, authenticate, login
from datetime import datetime
from home.models import Contact
from django.http import JsonResponse
import os
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.stem import WordNetLemmatizer
import nltk

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import BertTokenizer, BertForSequenceClassification
nltk.download('wordnet')
os.environ['OPENAI_API_KEY'] =  ""

import openai
# Load NLTK WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the fine-tuned model and tokenizer (replace 'fine_tuned_model' with actual path)
model_name = r"C:\Users\dell\OneDrive\Desktop\New folder (2)\fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\dell\OneDrive\Desktop\New folder (2)\fine_tuned_model\tokenizer")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def stringify_list_items(my_list):
    if not my_list:  # Check if the list is empty
        return 'NONE'  # Return 'NONE' if the list is empty
    else:
        return ''.join(str(item) if item != '' else 'NONE' for item in my_list)

# Function to lemmatize a word
def lemmatize_word(word):
    return lemmatizer.lemmatize(word, pos='v')  # 'v' indicates that the word is a verb

# Function to predict whether text is criminal or not and identify criminal words
def predict_and_identify_criminal(text):
    # Prepend [CLS] token and append [SEP] token
    text = "[CLS] " + text + " [SEP]"

    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)

    # Perform forward pass through the model
    outputs = model(**inputs)

    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Tokenize input text to get word indices
    tokens = tokenizer.tokenize(text)
    word_indices = tokenizer.convert_tokens_to_ids(tokens)

    # Identify words related to criminal activity
    criminal_words = ['abducted', 'abuse', 'ambush', 'break', 'dacoity', 'devise', 'devised', 'dispose', 'divert', 'fight', 'fire', 'fired', 'firing', 'fraud', 'gun', 'hide', 'kidnap', 'kidnapped', 'kidnapping', 'kill', 'killed', 'killing', 'loot', 'looted', 'murder', 'murdered', 'murdering', 'poisoned', 'raid', 'raided', 'rob', 'robbed', 'robbery', 'robbing', 'sexual', 'shoot', 'smuggled', 'smuggling', 'snatch', 'snatched', 'snatching', 'steal', 'stealing', 'stole', 'stolen', 'theft','threatened']

    # Lemmatize criminal words
    lemmatized_criminal_words = [lemmatize_word(word) for word in criminal_words]

    # Check if any criminal words are present in the input text
    criminal_present = any(token.lower() in lemmatized_criminal_words for token in tokens)

    # Construct output text with identified words
    output_text = ""
    criminal_found = []
    for i, token in enumerate(tokens):
        if word_indices[i] == tokenizer.cls_token_id:
            continue
        if word_indices[i] == tokenizer.sep_token_id:
            break
        # Lemmatize the token for comparison
        lemma_token = lemmatize_word(token.lower())
        if lemma_token in lemmatized_criminal_words:
            output_text += f"**{token}** "
            criminal_found.append(token)
        else:
            output_text += token + " "

    # Extract locations, dates, and times using regular expressions
    locations = re.findall(r'\b(?:in|at|on|near|from|to|into|through)\s([A-Za-z\s]+)\b', text)
    dates = re.findall(r'\b(\d{1,2}\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{2,4})\b', text, re.IGNORECASE)
    times = re.findall(r'\b(\d{1,2}(?::\d{2})?(?:\s?[ap]\.?m\.?))\b', text, re.IGNORECASE)

    # Return prediction along with input text, identified words, locations, dates, and times
    prediction_result = {
        "prediction": "This sentence contains criminal activity." if predicted_class == 1 else "This sentence does not contain criminal activity.",
        "criminal_words": criminal_found,
        "locations": locations,
        "dates": dates,
        "times": times,
        "output_text": output_text
    }
    return prediction_result

def loginuser(request):
    if request.method=="POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username, password)
        user = authenticate(username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect("/")

        else:
            return render(request, 'login.html')

    return render(request,'login.html')
def logoutuser(request):
    logout(request)
    return redirect("/login")

def index(request):
    username = request.user
    if request.user.is_anonymous:
        return redirect("/login")
    
    
    return render(request,'index.html',{'username': username})

def signup(request):
    if request.method=="POST":
        username=request.POST['username']
        email=request.POST['email']
        fname=request.POST['fname']
        lname=request.POST['lname']
        password=request.POST['password']
        password2=request.POST['password2']
        myuser = User.objects.create_user(username, email, password)
        myuser.first_name= fname
        myuser.last_name= lname
        myuser.save()
        return redirect('/login')
    return render(request,'signup.html')
def FileAudioForensic(request):
    username = request.user
    if request.user.is_anonymous:
        return redirect("/FileAudioForensic")
    return render(request,'FileAudioForensic.html',{'username': username})
def LiveAudioForensic(request):
    username = request.user
    if request.user.is_anonymous:
        return redirect("/LiveAudioForensic")
    return render(request,'LiveAudioForensic.html',{'username': username})
def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        desc = request.POST.get('desc')
        
        # Assuming you have a Contact model to save the data
        contact = Contact(name=name, email=email, phone=phone, desc=desc, date=datetime.today())
        contact.save()
        
        # Return a JSON response indicating success
        return JsonResponse({'message': 'Message sent successfully!'})
    else:
        # If it's not a POST request, return an empty response
        return JsonResponse({})
def test(request):
    return render(request,'test.html')
def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        with open('media/' + uploaded_file.name, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        lemmatizer = WordNetLemmatizer()
        folder_path = "E:\Fyp Project\Crime Watch\Criminal"
        file_list = os.listdir(folder_path)
        audio_files = [os.path.join(folder_path, file) for file in file_list if file.endswith(('.mp3', '.wav', '.ogg','.m4a'))]

        audio_file= open('media/' + uploaded_file.name, "rb")
        transcript = openai.Audio.translate("whisper-1", audio_file)
        text_to_predict= transcript['text']


        # Make prediction and identify criminal words, locations, dates, and times
        
        
        
        prediction_result = predict_and_identify_criminal(text_to_predict)


        # Print prediction and other extracted information
        context={
            "prediction":stringify_list_items(prediction_result["prediction"]),
            "Identifiedwords": stringify_list_items(prediction_result["criminal_words"]),
            "Locations": stringify_list_items(prediction_result["locations"]),
            "Dates":stringify_list_items(prediction_result["dates"]),
            "Times":stringify_list_items(prediction_result["times"]),
            "ProcessedText":text_to_predict
        }

        

        return JsonResponse(context)