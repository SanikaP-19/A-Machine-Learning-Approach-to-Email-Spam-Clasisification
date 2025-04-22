from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home page
    path('register/', views.register, name='register'),  # Register page
    path('login/', views.login_user, name='login'),  # Login page
    path('logout/',views.logout_user,name = 'logout'),
    path('dashboard/',views.dashboard,name='dashboard'),
    path('upload/',views.upload_dataset, name='upload_dataset'),
    path('datasets/', views.view_datasets, name='view_datasets'),  # View all datasets
    path('datasets/<int:dataset_id>/analyze/', views.analyze_dataset_view, name='analyze_dataset'),  # ML Analysis page
    path('history/', views.analysis_history, name='analysis_history'),  # Analysis history page
    path('classify_message/',views.classify_message_view,name='classify_message',)
]



    