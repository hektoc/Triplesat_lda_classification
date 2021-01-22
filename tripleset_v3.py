# from numba import jit
# import numpy as np
# import time

# @jit(parallel=True, cache=True)
def my_heavy_func(lda, img_data, num_start: int, pixel_in_row: int, shared_dict,
                  pixel_correct_coefficient: float) -> bool:
    """Тяжелая нагрузка, выполняющаяся в паралельных процессах.

    Принимаемые на вход параметры:
        lda - объект класса LdaClassificate
        img_data - одна строка изображения класса spectral
        num_start - начальный номер строки для записи в общий словарь
        pixel_in_row - длина строки, чтобы каждый раз не пересчитывать len(img_data)
        shared_dict - общий словарь, доступный между процессами
        pixel_correct_coefficient - коэффициент, на который, в случае необходимости, домножаются данные пикселов

    Возвращает True, вся полезная нагрузка возвращается через общий словарь shared_dict"""
    # start = time.time()
    # np.seterr(all='ignore')
    # цвета для классов берем из объекта lda
    colours = lda.get_lern_classes_names_and_colours()

    # объявляем функции для рассчёта коэффициентов. Требуется как костыль для переноса локали в другие процессы
    lda.indexes_compile()
    res = []
    for num in range(0, len(img_data)):
        for nel in range(pixel_in_row):  # pixel_in_row
            index_val = lda.find_indexes(b=(img_data[num][nel][0] * pixel_correct_coefficient),
                                         g=(img_data[num][nel][1] * pixel_correct_coefficient),
                                         r=(img_data[num][nel][2] * pixel_correct_coefficient),
                                         ir=(img_data[num][nel][3] * pixel_correct_coefficient))

            if index_val is not False:
                cat = lda.predict([index_val])
            else:
                cat = 'fail'
            res.append(colours[cat])

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
    from lda_classificate.LdaLearnClass import LdaLearnClass
    from PIL import Image, ImageFont, ImageDraw
    from progressbar.progressbar import print_progress_bar
    from multiprocessing import Pool, TimeoutError, Manager

    # Игнорируем ошибки в numpy
    np.seterr(all='ignore')

    '''Настройки исходных данные'''
    base_path = os.path.dirname(os.path.realpath(__file__))
    # Файл изображения # writeable=True
    img_data = spectral.open_image(base_path + r'\without_flaash\NNDiffusePanSharpening_cropped.hdr').open_memmap()
    # путь до папок с выборками
    lern_data_dir = base_path + r'\without_flaash\\'
    # корректирующий коэффициент, на который домножаются значения пикселов при классификации (но не обцчении)
    pixel_correct_coefficient = 1

    '''Настройки обработки'''
    # Каждый воркер обрабатывает по strings_per_worker строк
    strings_per_worker = 50
    # Сколько ядер процессора использовать для работы
    # workers_CPU_num = 4
    workers_CPU_num = os.cpu_count()  # максимально возможное значение
    # Если время обработки одного задания превысит worker_timeout секунд, обработка прекратится с ошибкой
    worker_timeout = 35

    # Если для тестов нужно переопределить размер изображения, то раскоментируем:
    # pixel_in_row = 500
    # img_data.rows_in_file = 40

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

    img_data.rows_in_file = len(img_data)
    pixel_in_row = len(img_data[0])

    results = []
    multiple_results = []

    # Создаём новое изображение, на котором будем отрисовывать результаты
    new_img = Image.new('RGB', (pixel_in_row, img_data.rows_in_file))

    lda = LdaClassificate()

    # Добавляем классы для классификации
    lda.add_learn_class(
        LdaLearnClass(name='1 - Почва', folder=lern_data_dir + 'pochva\\', colour=(139, 69, 19)))
    lda.add_learn_class(
        LdaLearnClass('2 - Подлесок', lern_data_dir + 'podlesok\\', (0, 255, 0)))
    lda.add_learn_class(
        LdaLearnClass('3 - Сухостой', lern_data_dir + 'suhostoj\\', (143, 188, 143)))
    lda.add_learn_class(
        LdaLearnClass('4 - Усыхающие деревья', lern_data_dir + 'usihanie\\', (128, 128, 0)))
    lda.add_learn_class(
        LdaLearnClass('5 - Здоровая растительность', lern_data_dir + 'zdorov_rast\\', (0, 100, 0)))

    lda.index_init(index_dict)
    lda.indexes_compile()
    lda.train()

    # стартуем workers_CPU_num воркеров в пуле процессов
    with Pool(processes=workers_CPU_num, ) as pool:
        # Используем менеджер для общения между потоками
        with Manager() as manager:
            # Для примера работы с менеджером создадим общий словарь, который будет доступен между процессами
            shared_dict = manager.dict()
            shared_timing = manager.dict()
            shared_timing['find_indexes'] = 0
            shared_timing['predict'] = 0
            shared_timing['shared_dict'] = 0
            shared_timing['all'] = 0

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
                    my_heavy_func, (lda,
                                    img_data[num_start:num_end],
                                    num_start,
                                    pixel_in_row,
                                    shared_dict,
                                    pixel_correct_coefficient,)))

                '''# Если решим запускаться без мультипроцессинга
                my_heavy_func(lda,
                                    img_data[num_start:num_end],
                                    num_start,
                                    pixel_in_row,
                                    shared_dict,
                                    pixel_correct_coefficient,)'''
                # multiple_results.append(pool.apply_async(print, (num_start,)))
                '''finished = (num_start + 1) / img_data.rows_in_file  # готовность в процентах
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
                # очищаем результаты, чтобы не висели в памяти. Они ведь очищаются?_)))
                multiple_results = []
            except TimeoutError:
                print("Превышен таймаут на операцию. Проверку можно пропустить")
            print(shared_timing['find_indexes'])
            print(shared_timing['predict'])
            print(shared_timing['shared_dict'])
            print(shared_timing['all'])
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

            # дорисовываем на изображении название классов
            draw = ImageDraw.Draw(new_img)
            font = ImageFont.truetype("tahoma.ttf", 16)
            pos = 0
            for name, colour in lda.get_lern_classes_names_and_colours().items():
                pos += 20
                draw.text((20, pos), name, colour, font=font, stroke_fill=(0, 0, 0), stroke_width=3)
            # сохраняем и показываем изображение
            new_img.save('./new_img.png')
            new_img.show()

    print("Выходим из with Pool, пул закрывается.")
