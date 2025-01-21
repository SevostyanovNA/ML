# ML
Системы управления и обработки информации (семинар наставника) ДЗ 1

Репозиторий содержит решение задачи CVRP для избранных датасетов (Set B, Set E, Set P http://vrp.atd-lab.inf.puc-rio.br/index.php/en/)

Решение представлено в файле Solution.py

Предыдущие версии сохранены под именами SaveX.py в качестве свидетельства развития изначального решения

Директории /X_params содержат набор гиперпараметров муравьиного алгоритма для каждой задачи, дающий наиболее качественное решение без затраты лишнего времени

Директории /vrp_files_X содержат файлы задач и решений (Дано)

## Инструкция к файлу решения
При запуске файла требуется выбрать режим работы из трех наборов опций:
### Choose datasets to process (e.g., e, p, b or all):
all - все датасеты

e, p, b - через запятую можно указать какие из наборов задач будут решены
### Use recommended parameters or tune new ones? (recommended/tune):
recommended - использовать подобранные гиперпараметры

tune - подобрать новые гиперпараметры (занимает долгое время, формирует файлы в /X_params заново)
### Choose output format: full, brief, or detailed (full/brief/detailed):
full - вывести в консоль решение в виде


  File: B-n34-k5.vrp

  Best Parameters: {'alpha': 2.0, 'beta': 2.0, 'evaporation_rate': 0.7, 'num_ants': 10, 'num_iterations': 500}
  
  Best Cost (Calculated): 804.0
  
  Optimal Value: 788
  
  Deviation: 2.03%


brief - вывести в консоль решение в виде


  File: B-n31-k5.vrp
  
  Best Cost (Calculated): 686.0
  
  Optimal Value: 672
  
  Deviation: 2.08%


detailed - вывести полную информацию, включая полученные маршруты
