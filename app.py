import sys
import os
from flask import render_template
from google.cloud import aiplatform
import vertexai
import requests
from vertexai.preview.language_models import TextGenerationModel
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])

def standardpage():
    if request.method == "POST":
## extract the values handed over from the form out of the POST request and store it in variables
        name1 = request.form['name1']
        name2 = request.form['name2']
        name3 = request.form['name3']
        place1 = request.form['place1']
        place2= request.form['place2']
        place3= request.form['place3']
        lesson= request.form['lesson']
## combine the variables in two larger strings
        characters = name1+","+name2+","+name3
        places = place1+","+place2+","+place3
## make the API call and store the result in a variable story
## first define the context using {placeholders} 
## handover the values for the placeholders in the input: section
        story= (predict_large_language_model_sample("<YOUR PROJECT ID>", "text-bison", 1, 582, 0.81, 40, 
        '''Create a short story for a child of age 4 to 10 which includes the characters {characters} and plays in the locations {locations} 
        and is about the lessons in life of {lessons}. Start the story with "Once upon a time..". End the story with "and now it is time to 
        go to sleep. Great that we could experience this story together. Talk to you tomorrow... Your bedtime stories team. The maximum output 
        length of the story should be 200 words in total. 
        input: characters = '''+characters+''' places = '''+places+''' lesson = '''+lesson+''' output:''', "us-central1"))  
## render the template return and handover the story as value which will be displayed        
        return render_template('return.html',value=story)

    else:    
        return render_template('startup.html')


# #  API call function

def predict_large_language_model_sample(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = ""
    ):
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name) 
    if tuned_model_name:
        model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
    content,
    temperature=temperature,
    max_output_tokens=max_decode_steps,
    top_k=top_k,
    top_p=top_p,)
    return response
    
    


        
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

sys.exit()


