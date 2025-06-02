import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Convertir etiquetas a números
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded
)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Guarda el modelo y el codificador de etiquetas
with open('../model/model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': le}, f)