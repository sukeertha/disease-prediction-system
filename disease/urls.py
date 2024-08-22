from django.urls import path
from .import views
urlpatterns=[
    path('',views.index),
    path('register/',views.register),
    path('saveregister/',views.saveregister),
    path('loginpage/',views.loginpage),
    path('checklogin/',views.checklogin),
    path('homepage/',views.homepage),
    path('lung_cancer/',views.lung_cancer),
    path('diabetes/',views.diabetes),
    path('heart_disease/',views.heart_disease),
    path('kidney_disease',views.kidney_disease),
    path('logout/',views.logout),
    path("check_lung_cancer/",views.check_lung_cancer),
    path("lung_result/<result>",views.lung_result,name="lung_result"),
    path('check_diabetes/',views.check_diabetes),
    path("diabetes_result/<result>",views.diabetes_result,name="diabetes_result"),
    path("check_kidney/",views.check_kidney),
    path("kidney_result/<result>",views.kidney_result,name="kidney_result"),
    path("check_heart_disease/",views.check_heart_disease),
    path("heart_disease_result/<result>",views.heart_disease_result,name="heart_disease_result"),
    path('about/',views.about),
    path('contact/',views.contact),

    path('savecontact/',views.savecontact),
    path('service/',views.service),


]