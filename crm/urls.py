from django.urls import path
from .views import home, model_performance, churn_factors, at_risk_customers, model_metrics, retrain_model

urlpatterns = [
    path('', home, name='home'),
    path('metrics/', model_metrics, name='model_metrics'),
    path('factors/', churn_factors, name='churn_factors'),
    path('at-risk/', at_risk_customers, name='at_risk_customers'),
    path('retrain/', retrain_model, name='retrain_model'),
]
