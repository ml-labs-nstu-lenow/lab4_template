import gradio as gr
import joblib
import numpy as np

from config import app_config

model = joblib.load(app_config.path_to_modelfile)


# Функция для инференса
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Преобразование входных данных в массив
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # Предсказание
    prediction = model.predict(input_data)
    # Возвращаем метку класса
    return prediction


# Создание интерфейса с gr.Blocks
with gr.Blocks(title="Инференс модели Iris") as demo:
    gr.Markdown("## 🌸 Прогнозирование вида ириса на основе признаков")

    with gr.Row():
        # Входные слайдеры для признаков
        # Можно определить текстовые поля для ввода
        sepal_length = gr.Slider(
            minimum=0, maximum=10, step=0.1, label="Длина чашелистика (см)"
        )
        sepal_width = gr.Slider(
            minimum=0, maximum=5, step=0.1, label="Ширина чашелистика (см)"
        )
        petal_length = gr.Slider(
            minimum=0, maximum=10, step=0.1, label="Длина лепестка (см)"
        )
        petal_width = gr.Slider(
            minimum=0, maximum=5, step=0.1, label="Ширина лепестка (см)"
        )

    # Кнопка для запуска инференса
    predict_btn = gr.Button("Предсказать вид")

    # Вывод результата
    result = gr.Textbox(label="Результат")

    # Связь кнопки и функции
    predict_btn.click(
        fn=predict_species,
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=result,
    )

# Запуск приложения
if __name__ == "__main__":
    demo.launch()
