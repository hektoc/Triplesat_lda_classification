class LdaClassificate:
    """
    Класс для проведения классификации методом LDA
    """
    import os

    def __init__(self, files_source: str = (os.path.dirname(os.path.realpath(__file__)) +
                                            "..\\flaash2_skip_some_points\\*.txt")):

        import lda_classificate.default_indexes
        import glob
        import numpy as np
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis


        print(files_source)

        np.seterr(all='ignore')

        self.default_indexes = lda_classificate.default_indexes.default_indexes
        self.indexes = {}
        files = glob.glob(files_source)
        self.lda_data = []
        lvl = []

        for name in files:
            # print(name)
            try:
                self.sp = np.genfromtxt(name, skip_header=1).T
            except:
                # нужно сделать обработку эксепшнов
                print(name, "SOME ERROR")
            lvl.append(int(np.genfromtxt(name).T[1][0]))
            self.lda_data.append(self.sp[1])

        data = np.array(self.lda_data)
        self.lda = LinearDiscriminantAnalysis(n_components=3)
        self.lda.fit(self.lda_data, lvl).transform(data)
        
        """self.lda = QuadraticDiscriminantAnalysis()
        # print(type(lvl), lvl)
        self.lda.fit(data, np.array(lvl))"""
        # exit()
        """from numpy import arange
        from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
        # define model
        model = LinearDiscriminantAnalysis(solver='lsqr')
        # define model evaluation method
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define grid
        grid = dict()
        grid['shrinkage'] = arange(0, 1, 0.01)
        # define search
        search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1, refit=True)
        # perform the search
        self.lda = search.fit(self.lda_data, lvl)
        # summarize
        print('Mean Accuracy: %.3f' % self.lda.best_score_)
        print('Config: %s' % self.lda.best_params_)"""

    def predict(self, indexes: list):
        # print(f'ind_calc({type(indexes)})={indexes},\n{type(indexes[0])}')
        return int(self.lda.predict(indexes)[0])  # int([new_sp])[0]

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

    def _index_count(self, index: dict, check_variable: bool = False, **kwargs):
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
            return eval(index['return'], locals(), kwargs)
        # Проверить возможные эксепшны и убрать этот позор
        except:
            print('Ошибка', sys.exc_info()[0])
            print('index=', index, '\nkwargs=', kwargs)
            # Функционал выпилен, но может пригодиться при модернизации
            if not check_variable:
                return self._index_count(index, True, kwargs)
            else:
                print(f'Ошибка пересчёта индекса {index}\nkwargs={kwargs}')
        return False

    def find_indexes(self, **kwargs):
        """Подсчитывает все выбранные индексы для выбранного 'пикселя'"""
        import numpy as np

        cur_indexes = []
        # print('find_indexes kwargs=', kwargs)
        # print(self.indexes)
        for name, index in self.indexes.items():
            # print('index=', index)
            index_value = self._index_count(index, False, **kwargs)
            # print(index_value)
            if np.isnan(index_value):
                cur_indexes.append(0)
                return False
            elif np.isinf(index_value):
                cur_indexes.append(10000)
            else:
                # if index_value <0:
                #    cur_indexes.append(0)
                # else:
                cur_indexes.append(index_value)
        return cur_indexes

