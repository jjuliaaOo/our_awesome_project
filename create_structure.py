import os

def create_project_structure():
    """Создает структуру папок проекта"""
    
    # Основные директории
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'notebooks',
        'src',
        'models/results',
        'reports/figures'
    ]
    
    # Файлы
    files = {
        '': ['requirements.txt', 'README.md', 'main.py'],
        'src/': ['__init__.py', 'data_processing.py', 'feature_extraction.py', 
                'models.py', 'visualization.py'],
        'notebooks/': ['01_data_loading.ipynb', '02_eda.ipynb', 
                      '03_feature_engineering.ipynb', '04_linear_models.ipynb',
                      '05_nonlinear_models.ipynb', '06_dnabert_analysis.ipynb']
    }
    
    # Создаем директории
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Создана папка: {directory}")
    
    # Создаем файлы
    for folder, file_list in files.items():
        for file in file_list:
            file_path = os.path.join(folder, file)
            with open(file_path, 'w') as f:
                # Добавляем базовое содержимое для некоторых файлов
                if file == 'requirements.txt':
                    f.write("""numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
biopython>=1.79
torch>=1.9.0
transformers>=4.20.0
plotly>=5.8.0
xgboost>=1.5.0
tqdm>=4.64.0
jupyter>=1.0.0
""")
                elif file == 'README.md':
                    f.write("# Классификация промоторных последовательностей кур\n\nПроект по биониформатике...")
                elif file.endswith('.py') and file != '__init__.py':
                    f.write(f'"""\nМодуль для {file.replace(".py", "")}\n"""\n\n')
            print(f"✓ Создан файл: {file_path}")
    
    print("\n🎉 Структура проекта создана!")

if __name__ == "__main__":
    create_project_structure()
