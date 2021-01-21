def my_heavy_func(lda, img_data, num_start: int, pixel_in_row: int, shared_dict):
    """Тяжелая нагрузка, выполняющаяся в паралельных процессах.

    Принимаемые на вход параметры:
        lda - объект класса LdaClassificate
        img_data - одна строка изображения класса spectral
        num_start - начальный номер строки для записи в общий словарь
        pixel_in_row - длина строки, чтобы каждый раз не пересчитывать len(img_data)
        shared_dict - общий словарь, доступный между процессами

    Возвращает True, вся полезная нагрузка возвращается через общий словарь"""
    import numpy as np
    # import time

    # print(pixel_in_row)
    np.seterr(all='ignore')
    # start = time.perf_counter()

    # цвета для классов
    COLOURS = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (128, 128, 0),
        3: (255, 0, 255),
        4: (255, 222, 173),
        5: (139, 69, 19),
        'fail': (255, 0, 0)
    }

    # print(f'Обработка {num_start}-{num_start+len(img_data)}')
    res = []
    for num in range(0, len(img_data)):
        # print(num)
        for nel in range(pixel_in_row):  # pixel_in_row
            index_val = lda.find_indexes(b=img_data[num][nel][0],
                                         g=img_data[num][nel][1],
                                         r=img_data[num][nel][2],
                                         ir=img_data[num][nel][3])
            if index_val is not False:
                cat = lda.predict([index_val])
            else:
                cat = 'fail'
            res.append(COLOURS[cat])
    shared_dict[num_start] = res
    # print(f"{os.getpid()}: Завершился за {round(finish - start)} сек.)"
    return True


if __name__ == '__main__':
    """
    Пример использования пула (Pool) процессов. Создается пул из PROCESSES_NUM процессов (задается ниже), 
    которые по очереди обрабатывают строки изображения.
    В примере также используется мэнеджер (Manager), позволяющий обмениваться данными между процессами. 
    Менеджер может создавать различные типы данных (list, dict, Namespace, Lock, RLock, Semaphore, BoundedSemaphore, 
    Condition, Event, Barrier, Queue, Value и Array. В примере используется Manager.dict().
    
    
    """
    import os
    import numpy as np
    import time
    import spectral
    from lda_classificate.LdaClassificate import LdaClassificate
    from PIL import Image
    from progressbar.progressbar import print_progress_bar
    from multiprocessing import Pool, TimeoutError, Manager

    np.seterr(all='ignore')
    base_path = os.path.dirname(os.path.realpath(__file__))
    results = []
    multiple_results = []

    # Каждый воркер обрабатывает по strings_per_worker строк
    strings_per_worker = 10
    # Сколько ядер процессора использовать для работы
    # workers_CPU_num = 4
    workers_CPU_num = os.cpu_count()  # максимально возможное значение
    # Если время обработки одного задания превысит worker_timeout секунд, обработка прекратится с ошибкой
    worker_timeout = 35

    # Файл изображения # writeable=True
    img_data = spectral.open_image(base_path + r'\without_flaash\NNDiffusePanSharpening_cropped.hdr').open_memmap()

    # Где находятся файлы для инициализации LDA
    lda_classificate_dir = base_path + r"\without_flaash\*.txt"

    img_data.rows_in_file = len(img_data)
    pixel_in_row = len(img_data[0])

    # Если для тестов нужно переопределить размер изображения, то раскоментируем:
    # pixel_in_row = 500
    # img_data.rows_in_file = 100

    new_img = Image.new('RGB', (pixel_in_row, img_data.rows_in_file))

    lda = LdaClassificate(files_source=lda_classificate_dir)

    # Задание индексов
    index_dict = {
        'MCARI2': True,
        'MTVI': True,
        'MTVI2': True,
        'SIPI': True,
        'LV': True,
        'BR': True,
        'GNDVI2': True,
        'DI1': True,
        'SIPI2': True,
        'NPCI': True,
        'BR625': True,
        'PSNDchla': True,
        'PSSRa': True,
        'PSSRc': True,
        'New2': True,
        'New13': True,
        # Пример задания пользовательского индекса
        # 'New14': {        'in_values': ['r', 'g', 'b'],        'return': 'r / b',    },
    }
    lda.index_init(index_dict)

    # стартуем workers_CPU_num воркеров в пуле процессов
    with Pool(processes=workers_CPU_num, ) as pool:
        # Используем менеджер для общения между потоками
        with Manager() as manager:
            # Для примера работы с менеджером создадим общий словарь, который будет доступен между процессами
            shared_dict = manager.dict()

            start_time = time.perf_counter()
            print(f'Первая строка из {img_data.rows_in_file}:')

            mas_stroka = []
            # Начинаем перебирать строки (от 0 до img_data.rows_in_file) с шагом strings_per_worker
            for num_start in range(0, img_data.rows_in_file, strings_per_worker):
                num_end = num_start+strings_per_worker
                print(f'Строка {num_start}-{num_end} / {img_data.rows_in_file}')
                """добавляем новое задание (pool.apply_async) в список multiple_results. 
                полезная нагрузка задания: 
                    my_haevy_func(lda, img_data[num_start:num_end], num_start, pixel_in_row, shared_dict,)"""
                multiple_results.append(pool.apply_async(
                    my_heavy_func, (lda, img_data[num_start:num_end], num_start, pixel_in_row, shared_dict,)))

                '''finished = (num + 1) / img_data.rows_in_file  # готовность в процентах
                now_time = time.perf_counter()
                all_time = (now_time - start_time) / finished
                print(f'({round(finished * 100, 5)}%)')
                print(f"Finished in {round(all_time, 2)} seconds")'''

            # Закрываем пул. Задания больше не принимаются
            pool.close()
            try:
                start = time.perf_counter()
                cur = 1
                all_res = len(multiple_results)

                # Перебираем задания и получаем результаты
                for res in multiple_results:
                    results.append(res.get(timeout=worker_timeout))
                    time_cur = time.perf_counter() - start

                    # print(f"Обработка длится уже {round(time_cur)} секунд.
                    # Всего продлится около {round(all*time_cur/cur)}")

                    # Отрисовываем прогрессбар
                    cur += 1
                    print_progress_bar(cur, all_res,
                                       prefix='Прогресс:',
                                       suffix=f"Обработка длится уже {round(time_cur)} сек.. Всего продлится около "
                                              f"{round(all_res*time_cur/cur)} сек..", length=50, print_end='\n')

                multiple_results = []
            except TimeoutError:
                print("Превышен таймаут на операцию. Проверку можно пропустить")

            # перебираем общий словарь, в который потоки писали результаты выполнения
            for num in range(img_data.rows_in_file+1):
                if num in shared_dict:
                    """Если строка с таким номером есть в общем словаре - 
                    достаём и добавляем в переменную, для записи в изображение"""
                    # print(f'{num} - присутствуеть, len={len(shared_dict[num])}')
                    mas_stroka.extend(shared_dict[num])
                # else:
                    # mas_stroka.extend((255, 0, 0))
            # Помещаем данные в изображение, сохраняем и показываем
            new_img.putdata(list(mas_stroka))
            new_img.save('./new_img.png')
            new_img.show()

    print("Выходим из with Pool, пул закрывается.")
