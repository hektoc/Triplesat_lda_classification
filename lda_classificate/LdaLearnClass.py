import glob

class LdaLearnClass:
    def __init__(self, name: str, folder: str, colour: tuple):
        self.wavelength = []
        self.sp = []
        self.sp_max = []
        self.name = name
        self.colour = colour

        for filename in glob.glob(folder+'*.txt'):
            self.read_envifile(filename)
        self.lerndata = {self.name: self.sp}
        self.sp_max = max(o for i in self.sp for o in i)

    def read_envifile(self, filename: str) -> None:
        with open(file=filename, mode="r") as file:
            sp_cur = []
            wavelength_cur = []
            for i, line in enumerate(file):
                if i > 2:  # Пропуск строк с заголовком
                    res = line.split()
                    sp_cur.append(float(res[1]))
                    wavelength_cur.append(float(res[0]))
            self.sp.append(sp_cur)
            self.wavelength.append(wavelength_cur)

    '''def __add__(self, other):
        self.lerndata.update(other.lerndata)
        return self'''


