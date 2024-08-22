import pandas as pd
from django.shortcuts import render, redirect
from.models import *

# Create your views here.
def index(request):
    return render(request,'index.html')
def register(request):
    return render(request,'register.html')
def loginpage(request):
    return render(request,'loginpage.html')
def homepage(request):
    return render(request,'homepage.html')
def lung_cancer(request):
    return  render(request,'lung_cancer.html')
def diabetes(request):
    return render(request,'diabetes.html')
def heart_disease(request):
    return render(request,'heart_disease.html')
def kidney_disease(request):
    return render(request,'kidney_disease.html')
def logout(request):
    return redirect("/")

def about(request):
    return  render(request,'about.html')
def contact(request):
    return render(request,'contact.html')
def service(request):
    return render(request,'service.html')
def saveregister(request):
    var=tbl_register()
    var.user_name=request.POST.get('username')
    var.email=request.POST.get('email')
    var.password=request.POST.get('psw')
    var.phone_number=request.POST.get('phone')
    var.save()
    return  redirect('/loginpage/')



def savecontact(request):
    var=tbl_contact()
    var.name=request.POST.get('name')
    var.email=request.POST.get('email')
    var.subject=request.POST.get('subject')
    var.message=request.POST.get('message')
    var.save()
    return redirect('/contact/')
def checklogin(request):
    if request.method=='POST':
        username=request.POST.get('username')
        password=request.POST.get('password')
        user=tbl_register.objects.filter(user_name=username,password=password)
        if user:
            us=tbl_register.objects.get(user_name=username,password=password)
            request.session['userid']=us.id
            return redirect('/homepage/')
        else:
            return redirect('/loginpage/')
from joblib import load
def check_lung_cancer(request):
    gender=request.POST.get("gender")
    if gender == "male":
        gen=1
    else:
        gen=0
    age=int(request.POST.get("age"))

    smoking=request.POST.get("smoking")
    if smoking=="Yes":
        smo=2
    else:
        smo=1
    yellowFingers=request.POST.get("yellowFingers")
    if yellowFingers=="Yes":
        yellow=2
    else:
        yellow=1

    anxiety=request.POST.get("anxiety")
    if anxiety == "Yes":
        anx=2
    else:
        anx=1
    peer_pressure=request.POST.get("peer_pressure")
    if peer_pressure == "Yes":
        peer=2
    else:
        peer=1
    chronic_disease=request.POST.get("chronic_disease")
    if chronic_disease=="Yes":
        chronic=2
    else:
        chronic=1

    fatigue=request.POST.get("fatigue")
    if fatigue=="Yes":
        fat=2
    else:
        fat=1
    allergy=request.POST.get("allergy")
    if allergy=="Yes":
        alle=2
    else:
        alle=1
    weezing=request.POST.get("weezing")
    if weezing =="Yes":
        wee=2
    else:
        wee=1
    alcohol_consuming=request.POST.get("alcohol_consuming")
    if alcohol_consuming=="Yes":
        alcohol=2
    else:
        alcohol=1
    coughing=request.POST.get("coughing")
    if coughing=="Yes":
        cough=2
    else:
        cough=1
    shortness_of_breath=request.POST.get("shortness_of_breath")
    if shortness_of_breath=="Yes":
        short=2
    else:
        short=1
    swallowing_difficulty=request.POST.get("swallowing_difficulty")
    if swallowing_difficulty=="Yes":
        swall=2
    else:
        swall=1
    chest_pain=request.POST.get("chest_pain")
    if chest_pain=="Yes":
        chest=2
    else:
        chest=1
    model=load("lung_mode.pkl")

    n=[[gen,age,smo,yellow,anx,peer,chronic,fat,alle,wee,alcohol,cough,short,swall,chest]]

    df = pd.read_csv("survey lung cancer.csv")

    # In[28]:

    df.head()

    # In[29]:

    df.tail()

    # In[30]:

    df.shape

    # In[31]:

    df.info()

    # In[32]:

    df.describe().T

    # In[33]:

    df.isnull().sum()

    # # # Label Encoding
    # GENDER: male=1,female=0
    # LUNG_CANCER:yes=1,No=0
    # # In[35]:

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['GENDER'] = le.fit_transform(df['GENDER'])

    # In[36]:

    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])

    # In[38]:

    df.dtypes

    # # Selecting x and y

    # In[39]:

    x = df.iloc[:, 0:15]
    x

    # In[40]:

    y = df.iloc[:, -1]
    y

    # # Splitting the dataset

    # In[41]:

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

    # In[42]:

    x_train

    # # Standardization

    # In[43]:

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit_transform(x_train)
    sample_data_scaled = sc.transform(n)
    result=model.predict(sample_data_scaled)
    print(result)
    if result == [1]:
        result="Yes"
    else:
        result="No"
    return redirect("lung_result",result=result)

def lung_result(request,result):
    return render(request,"lung_result.html",{"result":result})




def check_diabetes(request):
    pregnancies=request.POST.get("pregnancies")
    glucose=request.POST.get("glucose")
    blood_pressure=request.POST.get("blood_pressure")
    skin_thickness=request.POST.get("skin_thickness")
    insulin=request.POST.get("insulin")
    bmi=request.POST.get("bmi")
    diabetes_pedigree_function=request.POST.get("diabetes_pedigree_function")
    age=request.POST.get("age")
    model=load('diabetes_model.pkl')

    df = pd.read_csv("D:\project\diabetes.csv")

    # In[4]:

    df.head()

    # In[5]:

    df.tail()

    # In[6]:

    df.sample(5)

    # In[7]:

    df.shape

    # In[8]:

    df.columns

    # In[9]:

    df.info()

    # df.describe().T

    # In[10]:

    df.dtypes

    # In[11]:

    df.isna().sum()

    # In[12]:

    df['Outcome'].value_counts()

    # # selecting independent and dependent variable

    # In[13]:

    x = df.iloc[:, 0:8]


    # In[14]:

    y = df.iloc[:, -1]

    # # splting data set

    # In[15]:

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)


    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    sample_data=[[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age]]
    sample_data_scaled = sc.transform(sample_data)
    result=model.predict(sample_data_scaled)
    if result == [1]:
        result = "Yes"
    else:
        result = "No"
    return redirect("diabetes_result",result=result)
def diabetes_result(request,result):
    return render(request,'diabetes_result.html',{"result":result})

def check_kidney(request):
    age=float(request.POST.get("age"))
    bp=float(request.POST.get("bp"))
    sg=float(request.POST.get("sg"))
    al=float(request.POST.get("al"))
    su=float(request.POST.get("su"))
    pc=request.POST.get("pc")
    if pc=="normal":
        pc=1
    else:
        pc=0
    pcc=request.POST.get("pcc")
    if pcc=="present":
        pcc=1
    else:
        pcc=0
    ba=request.POST.get("ba")
    if ba=="present":
        ba=1
    else:
       ba=0
    bgr=float(request.POST.get("bgr"))
    bu=float(request.POST.get("bu"))
    sc=float(request.POST.get("sc"))
    hemo=float(request.POST.get("hemo"))
    pcv=int(request.POST.get("pcv"))
    htn=request.POST.get("htn")
    if htn=="Yes":
        htn=1
    else:
        htn=0
    dm=request.POST.get("dm")
    if dm=="Yes":
        dm=1
    else:
        dm=0
    cad=request.POST.get("cad")
    if cad=="Yes":
        cad=1
    else:
        cad=0
    appet=request.POST.get("appet")
    if appet=="good":
        appet=0
    else:
        appet=1
    pe=request.POST.get("pe")
    if pe=="Yes":
        pe=1
    else:
        pe=0
    ane=request.POST.get("ane")
    if ane=="Yes":
        ane=1
    else:
        ane=0
    model=load('kidney_model.pkl')

    sample_data=[[age,bp,sg,al,su,pc,pcc,ba,bgr,bu,sc,hemo,pcv,htn,dm,cad,appet,pe,ane]]
    result=model.predict(sample_data)
    if result == [1]:
        result = "Yes"
    else:
        result = "No"
    return redirect("kidney_result", result=result)
def kidney_result(request,result):
    return render(request,'kidney_result.html',{"result":result})


def check_heart_disease(request):
    age=request.POST.get("age")
    sex=request.POST.get("sex")
    if sex=="Male":
        sex=1
    else:
        sex=0
    cp=request.POST.get("cp")
    if cp=="Typical Angina":
        cp=0
    elif cp=="Atypical Angina":
        cp=1
    elif cp=="Non Anginal Pain":
        cp=2
    else:
        cp=3
    trestbps=request.POST.get("trestbps")
    chol=request.POST.get("chol")
    fbs=request.POST.get("fbs")
    if fbs=="Yes":
        fbs=1
    else:
        fbs=0
    restecg=request.POST.get("restecg")
    if restecg=="Normal":
        restecg=0
    elif restecg=="Having ST-T wave abnormality":
        restecg=1
    else:
        restecg=2
    thalach=request.POST.get("thalach")
    exang=request.POST.get("exang")
    if exang=="I have experienced chest pain or discomfort during physical activity":
        exang= 1
    else:
        exang= 0
    oldpeak=request.POST.get("oldpeak")
    slope=request.POST.get("slope")
    if slope=="Upsloping":
        slope=0
    elif slope=="Flat":
        slope=1
    else:
        slope=2
    ca=request.POST.get("ca")
    thal=request.POST.get("thal")
    if thal=="No thalsemia":
        thal=0
    elif thal=="Normal":
        thal=1
    elif thal=="Fixed defect":
        thal=2
    else:
        thal=3
    model=load('heart_model.pkl')
    df = pd.read_csv("D:\project\heart.csv")

    # In[3]:

    df.head()

    # In[4]:

    df.tail()

    # In[5]:

    df.shape

    # In[6]:

    df.info()

    # In[7]:

    df.isna().sum()

    # In[8]:

    df.describe().T

    # In[9]:

    df.dtypes

    # In[10]:

    df['target'].value_counts()

    # # selecting dependent and independent variable

    # In[11]:

    x = df.iloc[:, 0:13]


    # In[12]:

    y = df.iloc[:, -1]


    # # splitting dataset into training & testing data set

    # In[13]:

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=101)





    # # Standardisation

    # In[15]:

    import warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    # In[16]:

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    sample_data=[[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
    sample_data_scaled=sc.transform(sample_data)
    result=model.predict(sample_data_scaled)
    if result == [1]:
        result = "Yes"
    else:
        result = "No"
    return redirect("heart_disease_result",result=result)
def heart_disease_result(request,result):
    return render(request,'heart_disease_result.html',{"result":result})







