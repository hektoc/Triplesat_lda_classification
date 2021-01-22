import lda_classificate.default_indexes
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import os
from lda_classificate.LdaLearnClass  import LdaLearnClass

class LdaClassificate:
    """
    Класс для проведения классификации методом LDA
    """
    def __init__(self):
        self.lern_classes = []
        self.indexes = {}
        self.colours = None
        self.sp_max_cor = None
        self.default_indexes = lda_classificate.default_indexes.default_indexes
        self.functions = []
        np.seterr(all='ignore')

        self.lda = LinearDiscriminantAnalysis(n_components=3)
        
        """self.lda = QuadraticDiscriminantAnalysis()
        # print(type(lvl), lvl)
        self.lda.fit(data, np.array(lvl))"""
        # exit()

    def predict(self, indexes: list):
        # print(f'ind_calc({type(indexes)})={indexes},\n{type(indexes[0])}')
        #return int(self.lda.predict(indexes)[0])  # int([new_sp])[0]

        return self.lda.predict(indexes)[0]  # int([new_sp])[0]

    # Добавляем обучающий класс
    def add_learn_class(self, added_class: LdaLearnClass):
        self.lern_classes.append(added_class)

    def get_lern_classes_names_and_colours(self):
        if self.colours is None:
            self.colours = { lern_class.name: lern_class.colour for lern_class in self.lern_classes}
        return self.colours

    def train(self):
        training_data = []
        target_values = []
        # class_num = 1
        '''sp_max = 1
        sp_max = max(i.sp_max for i in self.lern_classes)
        self.sp_max_cor = 255/sp_max
        if sp_max > 255:
            func = lambda x: int(x * self.sp_max_cor)
        else:
            func = lambda x: x
'''
        for learn_class in self.lern_classes:
            for pixel_data in learn_class.sp:
                target_values.append(learn_class.name)  # class_num
                training_data.append(self.find_indexes(b=pixel_data[0],
                                                       g=pixel_data[1],
                                                       r=pixel_data[2],
                                                       ir=pixel_data[3]))
            # class_num += 1
        self.functions = []
        self.lda.fit(training_data, target_values)
        # exit()
        # self.lda.fit(training_data, target_values)#.transform(training_data)

    def index_init(self, index_dict: dict):
        """Пример задания коэффициентов
        index_dict = {
                            'MCARI2': {
                               'in_values': ['r', 'g', 'b'],
                               'return': '1.5 * (2.5 * (r - g) - 1.3 * (r - b)) / '
                                         'math.sqrt((2 * r + 1) ** 2 - (6 * r - 5 * math.sqrt(g)) - 0.5)',
                            },
                            'MCARI3': {
                                'in_values': ['r', 'g', 'b'],
                                'return': '1.5 * (2.5 * (r - g) - 1.3 * (r - b)) / '
                                          'math.sqrt((2 * r + 1) ** 2 - (6 * r - 5 * math.sqrt(g)) - 0.5)',
                            },
                            'MTVI2': True,
            }"""
        for name, index in index_dict.items():
            # существующие индексы
            if type(index) is bool:
                if name in self.default_indexes:
                    self.indexes[name] = self.default_indexes[name]
                else:
                    print(f'Нет такого индекса {index}')
                    # raise AttributeError
            # Заданные пользовательские индексы
            elif type(index) is dict:
                # тут по-хорошему нужно сделать проверку синтаксиса,
                # чтобы в формулах не было индексов, не заданных во входных данных... но да ладно)
                self.indexes[name] = index
        #self.indexes_compile()

    def _index_count(self, index: dict, name: str, check_variable: bool = False, **kwargs):
        """
        Подсчитывает один индекс

        index = {
            'in_values': ['r', 'g', 'b'],
            'return': 'g+b',
        }
        *kwargs = {
            'r' = 1,
            'g' = 2,
            'b' = 3
        }"""
        import sys
        import numpy as np
        # тут пишем пакеты, необходимые для подсчёта индексов. Они передаются в eval через locals()
        import math

        for value in index['in_values']:
            # print(value)
            if value not in kwargs:
                raise AttributeError
            elif kwargs[value] < 0:
                # print(f'Заменяем ошибочное значение {value}{kwargs[value]} на 0')
                kwargs[value] = np.float64(0)

        try:
            import math
            #print(self.functions[name])
            result = self.functions[name](b=kwargs['b'],
                                         g=kwargs['g'],
                                         r=kwargs['r'],
                                         ir=kwargs['ir'])
            # print(result)
            return result
            return eval(index['return'], locals(), kwargs)
        # Проверить возможные эксепшны и убрать этот позор
        except:
            print('Ошибка', sys.exc_info()[0])
            print('index=', index, '\nkwargs=', kwargs)
            # Функционал выпилен, но может пригодиться при модернизации
            if not check_variable:
                return self._index_count(index, name, True, kwargs)
            else:
                print(f'Ошибка пересчёта индекса {index}\nkwargs={kwargs}')
        return False

    def find_indexes(self, **kwargs):
        """Подсчитывает все выбранные индексы для выбранного 'пикселя'"""
        import numpy as n
        cur_indexes = []
        for name, index in self.indexes.items():
            index_value = self._index_count(index, name, False, **kwargs)
            if np.isnan(index_value):
                cur_indexes.append(0)
                return False
            elif np.isinf(index_value):
                cur_indexes.append(10000)
            else:
                cur_indexes.append(index_value)
        return cur_indexes

    def indexes_compile(self):

        import math
        code = ''
        for name, val in self.indexes.items():
            # 'def {name}({",".join(str(i) for i in val["in_values"])}): ' \
            code += \
                f"""
import math
#from numba import jit
#@jit(nopython=True, parallel=True)
def {name}(b,g,r,ir): 
    # import math
    global math
    return {val["return"]}\n"""
        #print(code)
        # exec(code)
        exec(code, globals(), locals())
        self.functions = locals()
        #print('Компиляция прошла')
