from django.utils import timezone

from django.shortcuts import render,redirect,get_object_or_404
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import DatasetUploadForm, MessageClassificationForm
from .models import Dataset
from .ml_utils import analyze_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
# Create your views here.
#Registration
def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
    
        if password1 == password2:
            if User.objects.filter(username=username).exists():
             messages.error(request,'Username already exists')
            else:
                user = User.objects.create_user(username=username,password=password1)
                user.save()
                messages.success(request,'Account created successfully')
                return redirect('login')
        else:
             messages.error(request,'Passwords do not match')
    return render(request,'register.html') 

#Login view
def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        user = authenticate(request,username = username,password = password)
        if user is not None:
            login(request,user)
            return redirect('view_datasets')#Redirect to dashboard or homepage
        else:
            messages.error(request,'Invalid credentials')
    return render(request,'login.html')

#Logout view
def logout_user(request):
    logout(request)
    return redirect('login')        

@login_required
def dashboard(request):
    return render(request,'dashboard.html')    

@login_required
def home(request):
    return render(request, 'home.html')

@login_required
def upload_dataset(request):
    form = DatasetUploadForm(request.POST,request.FILES)
    if form.is_valid():
        dataset = form.save(commit=False)
        dataset.user = request.user #Link dataset to logged-in user
        dataset.save()  
        messages.success(request,'DAtaset uploaded successfully!')
        return redirect('view_datasets') #Redirect to the dataset list
    else:
        form = DatasetUploadForm()
    return render(request,'upload_dataset.html',{'form':form})              

@login_required
def view_datasets(request):
    datasets = Dataset.objects.filter(user = request.user) #Only show dataset to logged-in users
    return render(request,'view_datasets.html',{'datasets':datasets})
@login_required
def analyze_dataset_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)

    if request.method == 'POST':
        algorithm = request.POST['algorithm']
        result = analyze_dataset(dataset.file.path, algorithm)

        if 'error' not in result:
            # Save analysis to history
            dataset.analysis_history.append({
                'algorithm': algorithm,
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'date': timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            dataset.save()
            messages.success(request, f"Analysis complete using {algorithm}!")
        else:
            messages.error(request, result['error'])

        return render(request, 'analyze_dataset.html', {
            'dataset': dataset,
            'result': result
        })

    # Always return something even if it's a GET request
    return render(request, 'analyze_dataset.html', {'dataset': dataset})

@login_required
def analysis_history(request):
    datasets = Dataset.objects.filter(user = request.user)
    return render(request,'analysis_history.html',{'datasets':datasets})            

def classify_message_view(request):
    result = None
    model = None  # To store the model after it has been trained

    if request.method == 'POST':
        form = MessageClassificationForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['message']
            
            # Assuming your dataset is located at a specific path
            dataset_path = 'path/to/your/dataset.csv'  # Replace with the actual path
            algorithm = 'Naive Bayes'  # You can make this dynamic, based on user input
            
            # Get model and analysis results from the dataset
            analysis_result = analyze_dataset(dataset_path, algorithm)
            if 'error' in analysis_result:
                result = {'error': analysis_result['error']}
            else:
                # Retrieve the trained model from the analysis result
                model = analysis_result['model']
                
                # Preprocess the message using the same vectorizer used during training
                tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
                X_message = tfidf.fit_transform([message])  # You may want to fit_transform once and save the fitted vectorizer

                # Make the prediction using the trained model
                prediction = model.predict(X_message)
                label = 'Spam' if prediction[0] == 1 else 'Ham'
                result = {'label': label}

    else:
        form = MessageClassificationForm()

    return render(request, 'classify_message.html', {'form': form, 'result': result})
