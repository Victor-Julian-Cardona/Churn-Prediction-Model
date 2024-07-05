# Generated by Django 5.0.6 on 2024-07-05 19:47

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Customer',
            fields=[
                ('customerID', models.CharField(max_length=255, primary_key=True, serialize=False)),
                ('gender', models.CharField(max_length=255)),
                ('SeniorCitizen', models.IntegerField()),
                ('Partner', models.CharField(max_length=255)),
                ('Dependents', models.CharField(max_length=255)),
                ('tenure', models.IntegerField()),
                ('PhoneService', models.CharField(max_length=255)),
                ('MultipleLines', models.CharField(max_length=255)),
                ('InternetService', models.CharField(max_length=255)),
                ('OnlineSecurity', models.CharField(max_length=255)),
                ('OnlineBackup', models.CharField(max_length=255)),
                ('DeviceProtection', models.CharField(max_length=255)),
                ('TechSupport', models.CharField(max_length=255)),
                ('StreamingTV', models.CharField(max_length=255)),
                ('StreamingMovies', models.CharField(max_length=255)),
                ('Contract', models.CharField(max_length=255)),
                ('PaperlessBilling', models.CharField(max_length=255)),
                ('PaymentMethod', models.CharField(max_length=255)),
                ('MonthlyCharges', models.FloatField()),
                ('TotalCharges', models.FloatField()),
                ('Churn', models.CharField(max_length=255)),
            ],
        ),
    ]
