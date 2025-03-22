from preprocess import TextPreprocessor
from train import ModelTrainer

# Preprocesar texto
text_preprocessor = TextPreprocessor()
predictors, label = text_preprocessor.load_and_preprocess_text("E:/Universidad/DeepLearning/Germana.txt")

# Entrenar modelo
model_trainer = ModelTrainer()
model_trainer.train_model(predictors, label, text_preprocessor.total_words, text_preprocessor.max_sequence_len)

# Cargar modelo y generar texto
model_trainer.load_model()
generate_text = model_trainer.generate_text("Hola, voy a", 10, text_preprocessor.max_sequence_len, text_preprocessor.tokenizer)
print(generate_text)
