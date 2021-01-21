import numpy as np
import time
from spectral import *
from lda_classificate.LdaClassificate import LdaClassificate

"""Скрипт для проведения классификации методом LDA для изображения Кайковского леса TripleSat
файлы ...IndexLevels.txt (рассчитанные в программе TripleSat_vyborka.py) используются как входные данные 
обуч. выборки для алгоритма LDA для построения карты классификации по всему изображению TripleSat"""

np.seterr(all='ignore')
# results = []
# processes_pool = []

# d = "E:/!data/FSR02/FSR_poleti_kajkovo_180620/polet2/indexes_for_LDA_TripleSat/flaash2_skip_some_points/"

# img = open_image('ENVI_4bands_cropped_new.hdr') старое фото с минусовыми КСЯ
img = open_image('C:/Users/Администратор/PycharmProjects/Zaja/flaash2/flaash2.hdr')
img_data = img.open_memmap(writeable=True)

rows_in_file = len(img_data)
pixel_in_row = len(img_data[0])
all_num = rows_in_file * pixel_in_row

lda = LdaClassificate()
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
    # 'New14': {        'in_values': ['r', 'g', 'b'],        'return': 'r / b',    },
}
lda.index_init(index_dict)

start_time = time.perf_counter()
print(f'Первая строка из {rows_in_file}:')

from PIL import Image

new_img = Image.new('RGB', (pixel_in_row, rows_in_file))

COLOURS = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (128, 128, 0),
    3: (255, 0, 255),
    4: (255, 222, 173),
    5: (139, 69, 19),
    'fail': (255, 0, 0)
}

with open("all_Kajk_.txt", 'w') as output_file:
    output_file.write(
        "Row_number\t" + "Column_number\t" + "Type_veg\t" + "1- Zdorov, 2- Podlesok, 3 - usihanie, 4 - suhostoj, 5 - pochva" + '\n')
    mas_stroka = []
    for num in range(rows_in_file):

        print(f'Строка {num} / {rows_in_file}')

        show_img_data = list

        for nel in range(pixel_in_row):
            index_val = lda.find_indexes(b=img_data[num][nel][0]/10000,
                                         g=img_data[num][nel][1]/10000,
                                         r=img_data[num][nel][2]/10000,
                                         ir=img_data[num][nel][3]/10000)
            # print(index_val)
            if index_val is not False:
                cat = lda.predict([index_val])
            else:
                # print('cat=0', num, nel)
                cat = 'fail'
            mas_stroka.append(COLOURS[cat])
            #output_file.write(str(num) + "\t" + str(nel) + "\t" + str(cat) + "\n")
            #output_file.write("\n")
            #img_data.append()


        finished = (num + 1) / rows_in_file  # готовность в процентах
        now_time = time.perf_counter()
        all_time = (now_time - start_time) / finished
        print(f'({round(finished * 100, 5)}%)')
        print(f"Finished in {round(all_time,2 )} seconds")

        if num % 100 == 0 and nel == pixel_in_row - 1:
            new_img.putdata(list(mas_stroka))
            # mas_stroka = []
            new_img.show()
            new_img.save('./new_img.png')

new_img.show()

if __name__ == '__main__':
    pass