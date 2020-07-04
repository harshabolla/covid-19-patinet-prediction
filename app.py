from flask import Flask,render_template,request
import pickle

import numpy as np

app = Flask(__name__)

filename = 'corona.pkl'
clf = pickle.load(open(filename, 'rb'))



@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method =='POST':
        
        #pickle_in = open("corona.pkl","rb")
        
        					
        
        age = int(request.form['Age'])
        fever = int(request.form['Fever'])
        body = int(request.form['BodyPains'])
        cold = int(request.form['RunnyNose'])
        breath= int(request.form['Difficulty_in_Breath'])
        
         
        
        data = np.array([[age,fever,body,cold,breath]])

        #print('#-------------------------------data is here-------------------------------------#')
        #print(age,body,fever,cold,breath)
        
        #clf = pickle.load(open("corona.pkl", "rb"))
        #columns  -- Age,	Fever,	BodyPains,	RunnyNose,	Difficulty_in_Breath
        #data = [[int(age),int(fever),int(body),int(cold),int(breath)]]
        my_prediction = clf.predict(data)
        proba_score = clf.predict_proba([[60,100,0,1,0]])[0][0]
        
        #if predict==1:
            #prediction='Positive'
        #else:
           # prediction = 'Negative'
        
        #return render_template('index.html',prediction=prediction,proba_score=round(proba_score*100,2))
        return render_template('result.html', prediction=my_prediction, proba_score=round(proba_score*100,2))
    
    else:
        
        return render_template('index.html',message='Something missed, Please follow the instructions..!')
              

if __name__ == '__main__':
    app.run(debug=True)