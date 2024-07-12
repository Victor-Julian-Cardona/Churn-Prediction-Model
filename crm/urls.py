from django.urls import path
from .views import home, model_metrics, churn_factors, at_risk_customers

urlpatterns = [
    path('', home, name='home'),
    path('metrics/', model_metrics, name='model_metrics'),
    path('factors/', churn_factors, name='churn_factors'),
    path('at-risk/', at_risk_customers, name='at_risk_customers'),
]
