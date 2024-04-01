from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import torch
import joblib
import os
from network import SimpleCNN, AttentionClassifier
from src.save_and_load_model import save_model, load_model

app = Flask(__name__)

# 模型文件路径
MODEL_PATHS = {
    'RandomForest': 'best_model_RandomForest.joblib',
    'SVM': 'best_model_SVM.joblib',
    'GaussiaNB': 'best_model_GaussianNB.joblib',
}

# 预加载机器学习模型
ML_MODELS = {model: joblib.load(path) for model, path in MODEL_PATHS.items() if 'NeuralNetwork' not in model}


# 函数：加载神经网络模型
def load_neural_network_model():
    model = AttentionClassifier(num_features=470, num_classes=5)
    load_model(model)
    model.eval()
    return model


# 神经网络模型
NN_MODEL = load_neural_network_model()

# 类别名称
CLASS_NAMES = {
    0: 'Advertising software',
    1: 'Bank malware',
    2: 'SMS malware',
    3: 'Risk software',
    4: 'Normal'
}
label_names = {1: 'Advertising software', 2: 'Bank malware', 3: 'SMS malware', 4: 'Risk software', 5: 'Normal'}

@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'model_selection' not in request.form:
        return 'Missing file or model selection'

    file = request.files['file']
    selected_model = request.form['model_selection']
    print(selected_model)
    if file.filename == '':
        return 'No selected file'

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join('/path/to/save/uploads', filename)
        file.save(filename)

        data = pd.read_csv(filename)
        results = []

        if selected_model in ML_MODELS:
            clf = ML_MODELS[selected_model]
            X = data.drop('Class', axis=1).values
            y_pred = clf.predict(X)
            for index, (prediction, actual_label) in enumerate(zip(y_pred, data['Class'])):
                predicted_class = CLASS_NAMES[prediction]
                actual_class = label_names[actual_label]
                results.append((index, predicted_class, actual_class))
        elif selected_model == 'NeuralNetwork':
            for index, row in data.iterrows():
                features = torch.tensor(row.drop('Class').values.astype(float)).float().unsqueeze(0)
                with torch.no_grad():
                    output = NN_MODEL(features)[0]
                    prediction = output.argmax(1).item()
                predicted_class = CLASS_NAMES[prediction]
                actual_class = label_names[row['Class']]
                results.append((index, predicted_class, actual_class))
        else:
            return 'Selected model not found'

        return render_template('results.html', results=results)

    return 'Invalid file format'


if __name__ == '__main__':
    app.run(debug=True)
